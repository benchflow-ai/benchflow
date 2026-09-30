"""Training support: rollout groups against a trainer's policy, and launch helpers.

``bf.Policy`` and ``bf.rollout_group`` (:mod:`benchflow.training.rollouts`)
are the trainer-facing rollout API; :mod:`benchflow.training.relay` is the
authenticated relay that carries rollouts' model calls to the policy server.
"""
