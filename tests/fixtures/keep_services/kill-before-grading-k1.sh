P=/proc
uid=$(id -u agent 2>/dev/null) || { echo 'Invalid sandbox UID' >&2; exit 2; }
case "$uid" in ''|*[!0-9]*|0) echo 'Invalid sandbox UID' >&2; exit 2 ;; esac
if [ ! -r "$P/self/status" ]; then
    echo 'Process table is unavailable' >&2
    exit 2
fi
gid=$(id -g agent 2>/dev/null)
case "$gid" in ''|*[!0-9]*) gid=$uid ;; esac
me=
{ while read -r key first second rest || [ -n "$key" ]; do
    [ "$key" = Uid: ] && me=$second
    key=
done <"$P/self/status"; } 2>/dev/null
if [ "$me" != 0 ] && [ 1 != 0 ]; then
    echo 'The kill before grading is not running as root' >&2
    exit 2
fi
# The descriptor inspections in the keeping/release paths need a helper
# that can read as the user too. Signals still check each actual invocation.
as_user=
can_be() (
    check="[ \"\$(id -u)\" = $uid ] && [ \"\$(id -g)\" = $gid ]"
    case "$1" in
        setpriv) setpriv --reuid="$uid" --regid="$gid" --clear-groups -- sh -c "$check" </dev/null >/dev/null 2>&1 ;;
        su) su -s /bin/sh agent -c "$check" </dev/null >/dev/null 2>&1 ;;
    esac
)
if [ "$me" = 0 ]; then
    for helper in setpriv su; do
        if command -v "$helper" >/dev/null 2>&1 && can_be "$helper"; then
            as_user=$helper
            break
        fi
    done
fi
asuser() {
    if [ "$me" != 0 ]; then
        eval "$1" >/dev/null
        return 0
    fi
    body="[ \"\$(id -u)\" = $uid ] && [ \"\$(id -g)\" = $gid ] || exit 3; $1"
    for helper in setpriv su; do
        command -v "$helper" >/dev/null 2>&1 || continue
        case "$helper" in
            setpriv) out=$(setpriv --reuid="$uid" --regid="$gid" --clear-groups -- sh -c "$body" </dev/null 2>/dev/null) ;;
            su) out=$(su -s /bin/sh agent -c "$body" </dev/null 2>/dev/null) ;;
        esac
        if [ "$out" = 'BF as-user' ]; then
            as_user=$helper
            return 0
        fi
    done
    return 1
}
helpless() {
    echo "Cannot send SIG$1 as the sandbox user: neither setpriv nor su ran it as UID $uid GID $gid (install setpriv or su in the image and let root switch to that UID and GID)" >&2
    exit 2
}
everyone() {
    [ "$me" = 0 ] || return 0
    # kill(-1) runs in the helper's own shell, which it skips; in a ``$( )``
    # subshell it would stop that shell, still waiting for it. Its error, if
    # any, goes to the output root reads. A child gives even an otherwise
    # empty UID a recipient: ESRCH never has to count as success. Keep its
    # pid as our unreaped child through cleanup, and signal it as the user.
    asuser "sleep 60 </dev/null >/dev/null 2>&1 & probe=\$!; delivered=; if LC_ALL=C kill -$1 -1 2>&1; then delivered=1; fi; kill -KILL \"\$probe\" 2>/dev/null; wait \"\$probe\" 2>/dev/null; [ \"\$delivered\" = 1 ] && echo BF as-user" || helpless "$1"
}
each() {
    sig=$1
    shift
    mine=
    for pid in "$@"; do
        case "$pid" in
            r*) kill -s "$sig" "${pid#r}" 2>/dev/null ;;
            *) mine="$mine $pid" ;;
        esac
    done
    [ -n "$mine" ] || return 0
    asuser "for p in$mine; do kill -s $sig \$p 2>/dev/null; done; echo BF as-user" || helpless "$sig"
}
live() {
    for entry in "$P"/[0-9]*; do
        pid=${entry##*/}
        case "$pid" in ''|*[!0-9]*) continue ;; esac
        state=
        ruid=
        euid=
        suid=
        threads=
        if { while read -r key first second third rest || [ -n "$key" ]; do
                case "$key" in
                    State:) state=$first ;;
                    Uid:) ruid=$first euid=$second suid=$third ;;
                    Threads:) threads=$first ;;
                esac
                key=
                # Stop once the fields the test below reads are in (Threads
                # only for a zombie): the shell reads the file a byte at a
                # time, and the ~45 lines after Uid made each walk cost
                # ~15 ms in a trial's container (PERF-G).
                if [ -n "$state" ] && [ -n "$euid" ] &&
                    { [ "$state" != Z ] || [ -n "$threads" ]; }; then
                    break
                fi
            done <"$entry/status"; } 2>/dev/null &&
            [ -n "$state" ] && [ -n "$euid" ]; then
            tag=
            if [ "$ruid" != "$uid" ] && [ "$suid" != "$uid" ]; then
                [ "$euid" = "$uid" ] || continue
                tag=r
            fi
            if [ "$state" != Z ]; then
                echo "$tag$pid"
            else
                case "$threads" in ''|*[!0-9]*) ;; *) [ "$threads" -gt 1 ] && echo "$tag$pid" ;; esac
            fi
        elif [ -d "$entry" ]; then
            echo "Cannot inspect process $pid" >&2
            return 2
        fi
    done
    return 0
}
# Keeping uses numeric pids for socket and lineage lookups. Recheck their
# signal role here, then use the same UID-constrained delivery as the plain
# kill. Root's all-zero replacement is skipped; only the effective-UID-only
# population needs root's residual list/signal window.
signal_one() {
    sr= se= ssaved=
    { while read -r key a b c rest || [ -n "$key" ]; do
        [ "$key" = Uid: ] && { sr=$a; se=$b; ssaved=$c; }
        key=
    done <"$P/$2/status"; } 2>/dev/null || return 0
    if [ "$sr" = "$uid" ] || [ "$ssaved" = "$uid" ]; then
        each "$1" "$2"
    elif [ "$se" = "$uid" ]; then
        each "$1" "r$2"
    fi
    return 0
}
freeze() {
    everyone STOP
    stopped=' '
    pass=0
    while [ "$pass" -lt 100 ]; do
        pass=$((pass + 1))
        found=$(live) || exit 2
        new=
        for pid in $found; do
            case "$stopped" in *" $pid "*) continue ;; esac
            new="$new $pid"
            stopped="$stopped$pid "
        done
        [ -z "$new" ] && break
        each STOP $new
    done
    found=$(live) || exit 2
    for pid in $found; do
        # The root/user signaling role can change during the grace without
        # changing the process. Count its numeric pid only.
        pid=${pid#r}
        case "$killed" in *" $pid "*) ;; *) killed="$killed$pid "; n=$((n + 1)) ;; esac
    done
}
killed=' '
n=0
freeze
each TERM $found
each CONT $found
everyone CONT
if [ -n "$found" ]; then
    if sleep 0.1 2>/dev/null; then tick=0.1 ticks=10; else tick=1 ticks=1; fi
    t=0
    while [ "$t" -lt "$ticks" ]; do
        found=$(live) || exit 2
        [ -z "$found" ] && break
        sleep "$tick"
        t=$((t + 1))
    done
fi
round=0
while [ "$round" -lt 8 ]; do
    round=$((round + 1))
    freeze
    each KILL $found
    everyone KILL
    found=$(live) || exit 2
    if [ -z "$found" ]; then
        echo "BF killed $n end"
        exit 0
    fi
    sleep 1
done
found=$(live) || exit 2
if [ -z "$found" ]; then
    echo "BF killed $n end"
    exit 0
fi
names=
for pid in $found; do
    pid=${pid#r}
    cmd=$(tr '\000\n' ' ?' <"$P/$pid/cmdline" 2>/dev/null | cut -c1-200)
    names="$names $pid ($cmd)"
done
echo "Sandbox-user writers remain:$names" >&2
exit 1
