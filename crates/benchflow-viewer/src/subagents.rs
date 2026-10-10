//! Nest subagent steps under the tool call that spawned them (0.8's
//! `viewer/subagents.py`).
//!
//! Claude captures attribute a child agent's events to its spawning tool
//! call through `parent_tool_call_id`. An event belongs to the scope that
//! field names; a scope hangs under the scope holding the first captured
//! tool call with that id, found breadth-first from the main agent and at
//! most [`MAX_SUBAGENT_DEPTH`] levels deep (the ATIF export's rules). A
//! scope whose spawning call was never captured, or that cannot be reached
//! (cyclic attribution, or deeper than the cap), is shown as a labelled
//! group at the root, so every recorded event stays visible exactly once.

use std::collections::{HashMap, HashSet, VecDeque};

use indexmap::IndexMap;

use crate::steps::{Body, Event, Group, GroupReason, Source, Step, SubagentTrace};

/// How deep subagents nest (0.8's `export_atif.MAX_SUBAGENT_DEPTH`).
pub const MAX_SUBAGENT_DEPTH: usize = 8;

/// `None` is the main agent's scope.
type Scope = Option<String>;

/// Nest attributed steps; `steps` and `sources` are parallel, in capture order.
pub(crate) fn nest(steps: Vec<Step>, sources: Vec<Source>) -> Vec<Step> {
    debug_assert_eq!(steps.len(), sources.len());
    // Scopes in first-seen order (a Python dict), each with its step indexes.
    let mut scopes: IndexMap<Scope, Vec<usize>> = IndexMap::new();
    scopes.insert(None, Vec::new());
    let mut scope_of = Vec::with_capacity(sources.len());
    let mut call_ids: HashMap<usize, String> = HashMap::new();
    let mut spawns: HashMap<String, usize> = HashMap::new();
    for (index, (step, source)) in steps.iter().zip(&sources).enumerate() {
        scopes.entry(source.scope.clone()).or_default().push(index);
        scope_of.push(source.scope.clone());
        let is_tool = matches!(
            step,
            Step::Event(Event {
                body: Body::Tool(_),
                ..
            })
        );
        if let (true, Some(call_id)) = (is_tool, &source.call_id) {
            call_ids.insert(index, call_id.clone());
            spawns.entry(call_id.clone()).or_insert(index);
        }
    }
    if scopes.len() == 1 {
        return steps;
    }

    // Breadth-first from the main agent: which scopes are reachable, how
    // deep, and under which scope each one hangs.
    let mut depth: HashMap<Scope, usize> = HashMap::from([(None, 0)]);
    let mut children: HashMap<Scope, Vec<String>> = HashMap::new();
    let mut queue: VecDeque<Scope> = VecDeque::from([None]);
    while let Some(scope) = queue.pop_front() {
        let mut found: Vec<String> = scopes[&scope]
            .iter()
            .filter_map(|index| {
                let call_id = call_ids.get(index)?;
                (scopes.contains_key(&Some(call_id.clone())) && spawns[call_id] == *index)
                    .then(|| call_id.clone())
            })
            .collect();
        if scope.is_none() {
            // Scopes whose spawning call was never captured hang under the
            // main agent.
            found.extend(
                scopes
                    .keys()
                    .flatten()
                    .filter(|child| !spawns.contains_key(*child))
                    .cloned(),
            );
        }
        let scope_depth = depth[&scope];
        for child in found {
            let key = Some(child.clone());
            if depth.contains_key(&key) || scope_depth >= MAX_SUBAGENT_DEPTH {
                continue;
            }
            depth.insert(key.clone(), scope_depth + 1);
            children.entry(scope.clone()).or_default().push(child);
            queue.push_back(key);
        }
    }

    let spawn_sources: HashMap<String, Source> = spawns
        .iter()
        .map(|(call_id, index)| (call_id.clone(), sources[*index].clone()))
        .collect();
    let mut nester = Nester {
        slots: steps.into_iter().map(Some).collect(),
        scopes,
        scope_of,
        call_ids,
        spawns,
        spawn_sources,
        depth,
        children,
        groups: 0,
    };
    nester.build(&None)
}

struct Nester {
    /// Steps not placed yet, by capture index.
    slots: Vec<Option<Step>>,
    scopes: IndexMap<Scope, Vec<usize>>,
    scope_of: Vec<Scope>,
    call_ids: HashMap<usize, String>,
    /// The first captured tool call with each id.
    spawns: HashMap<String, usize>,
    spawn_sources: HashMap<String, Source>,
    depth: HashMap<Scope, usize>,
    children: HashMap<Scope, Vec<String>>,
    groups: usize,
}

impl Nester {
    fn take(&mut self, index: usize) -> Step {
        self.slots[index].take().expect("each step is placed once")
    }

    /// The steps of one scope, with nested scopes attached to their spawns.
    fn build(&mut self, scope: &Scope) -> Vec<Step> {
        let spawned: HashSet<String> = self
            .children
            .get(scope)
            .into_iter()
            .flatten()
            .cloned()
            .collect();
        let mut placed: Vec<(usize, Step)> = Vec::new();
        for index in self.scopes[scope].clone() {
            let mut step = self.take(index);
            // The first captured call with an id that names a reachable
            // scope spawned it: attach that scope's steps to the call.
            if let Some(call_id) = self.call_ids.get(&index).cloned()
                && spawned.contains(&call_id)
                && self.spawns[&call_id] == index
            {
                let nested = self.build(&Some(call_id.clone()));
                let trace = self.trace(&call_id, nested);
                if let Step::Event(Event {
                    body: Body::Tool(tool),
                    ..
                }) = &mut step
                {
                    tool.subagent = Some(trace);
                }
            }
            placed.push((index, step));
        }
        if scope.is_none() {
            // Groups sit at the capture position of their first event.
            let orphans: Vec<String> = self
                .children
                .get(&None)
                .into_iter()
                .flatten()
                .filter(|child| !self.spawns.contains_key(*child))
                .cloned()
                .collect();
            for child in orphans {
                let key = Some(child.clone());
                let position = self.scopes[&key][0];
                let nested = self.build(&key);
                let group = self.group(&child, GroupReason::ParentNotCaptured, nested);
                placed.push((position, group));
            }
            let unreachable: Vec<(String, Vec<usize>)> = self
                .scopes
                .iter()
                .filter_map(|(key, indexes)| {
                    let name = key.as_ref()?;
                    (!self.depth.contains_key(key)).then(|| (name.clone(), indexes.clone()))
                })
                .collect();
            for (orphan, indexes) in unreachable {
                let reason = self.unreachable_reason(&orphan);
                let nested = indexes.iter().map(|index| self.take(*index)).collect();
                let group = self.group(&orphan, reason, nested);
                placed.push((indexes[0], group));
            }
        }
        placed.sort_by_key(|(index, _)| *index);
        placed.into_iter().map(|(_, step)| step).collect()
    }

    fn trace(&self, scope: &str, steps: Vec<Step>) -> SubagentTrace {
        let spawn = self.spawn_sources.get(scope);
        SubagentTrace {
            parent_tool_call_id: scope.to_string(),
            depth: self
                .depth
                .get(&Some(scope.to_string()))
                .copied()
                .unwrap_or(0),
            steps,
            subagent_type: spawn.and_then(|source| source.subagent_type.clone()),
            description: spawn.and_then(|source| source.description.clone()),
        }
    }

    fn group(&mut self, scope: &str, reason: GroupReason, steps: Vec<Step>) -> Step {
        self.groups += 1;
        Step::Group(Group {
            gid: format!("g{}", self.groups),
            reason,
            subagent: self.trace(scope, steps),
        })
    }

    /// Why a scope is out of reach: its chain of spawning calls ends at a
    /// call that was never captured, loops, or reaches the main agent only
    /// beyond the depth cap.
    fn unreachable_reason(&self, scope: &str) -> GroupReason {
        let mut seen: HashSet<String> = HashSet::new();
        let mut current: Scope = Some(scope.to_string());
        while let Some(name) = current.clone() {
            if self.depth.contains_key(&current) {
                break;
            }
            let Some(spawn) = self.spawns.get(&name) else {
                return GroupReason::ParentNotCaptured;
            };
            if !seen.insert(name) {
                return GroupReason::Cyclic;
            }
            current = self.scope_of[*spawn].clone();
        }
        GroupReason::TooDeep
    }
}
