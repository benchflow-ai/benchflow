// Command generator writes ../go-shellwords-parse.jsonl: what Docker Compose
// makes of a service's `command` string, for the Rust port in
// crates/benchflow-task/src/shellwords.rs.
//
// Compose (compose-go v2, types/command.go, ShellCommand.DecodeMapstructure)
// splits a string command with shellwords.Parse, the package's default
// parser: ParseEnv, ParseBacktick and ParseComment are all false. go.mod pins
// go-shellwords v1.0.15, the version Docker Compose v5.6.0 builds in (`go
// version -m` on its docker-compose binary); v1.0.14 (Compose v5.4.0 to
// v5.5.1) differs only when ParseComment is on, and writes the same file.
//
// Each line is {"text": ..., "words": [...]} or {"text": ..., "error": ...}.
// Inputs, in this order and each once: the cases below, the "text" of every
// line of the JSONL files named as arguments, then 6,000 random strings from
// a fixed seed (its own generator, so every Go version writes the same file).
// The first run read the 2,567 strings of D5's Python shlex fixture (since
// removed); they are in the output, so to regenerate, feed it the fixture
// itself (Go 1.22 or later):
//
//	cd crates/benchflow-task/tests/fixtures/shellwords/generator
//	go run . ../go-shellwords-parse.jsonl > /tmp/parse.jsonl
//	mv /tmp/parse.jsonl ../go-shellwords-parse.jsonl
//
// The Rust test parse_matches_go_shellwords (src/shellwords.rs) compares
// every line.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/mattn/go-shellwords"
)

// Cases from the review of D5 and the shapes service commands take.
var cases = []string{
	"redis-server --save '' --requirepass #hunter2",
	"sh -c \"echo \\`date\\`\"",
	"sh -c \"grep -E 'a\\.b' f\"",
	"redis-server --save 60 1 > /dev/null",
	"python -c print(1)",
	"python -c 'print(1)'",
	"sh -c 'echo $(date)'",
	"echo $(date)",
	"echo \"$(date)\"",
	"echo \\$(date)",
	"echo a\\\\$(date)",
	"echo $(echo (x))",
	"echo $(a)b$(c)",
	"echo $( a b )",
	"'$'(x)",
	"'\\'$(x)",
	"echo )",
	"echo a; echo b",
	"a&&b",
	"a|b",
	"cmd < in.txt",
	"cmd 2>&1",
	"cmd 2> err.log",
	"cmd 12>x",
	"cmd a2>x",
	"cmd \"2\"> x",
	"cmd >&2",
	"printf \"a\\tb\\n\"",
	"printf a\\tb",
	"echo 'a\\tb'",
	"echo \"a\\nb\" a\\nb 'a\\nb'",
	"sh -c \"echo \\\"\\$HOME\\\"\"",
	"echo $HOME ${USER:-x} $$",
	"echo `date`",
	"echo `echo \"a b\" 'c'`",
	"echo `a\\`b`",
	"echo `unclosed",
	"echo \"unclosed",
	"echo 'unclosed",
	"echo trailing\\",
	"''",
	"\"\"",
	"",
	"   ",
	"a\\ b",
	"a\\\nb",
	"postgres -c port=5432 -c 'max_connections=200'",
	"--requirepass \"#x\" #y",
	"nginx -g 'daemon off;'",
	"sh -c \"nginx -g 'daemon off;' && sleep 1\"",
	"sleep infinity",
	"bash -c 'while true; do sleep 1; done'",
	"python3 -m http.server 8000",
	"sh -c \"exec python -c 'print(1)'\"",
	"mysqld --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci",
	"java -Xmx512m -jar /app/app.jar --server.port=8080",
	"uvicorn app:app --host 0.0.0.0 --port 8000",
	"echo é 中 😀  x y",
	"a\x00b c",
	// The D5b review's cases: two literal backslashes before $( (a buffer
	// ending in \\$), then three; and two where Compose substitutes
	// variables before it splits ($$ becomes $, ${X:-$} becomes $), so
	// Compose's answer can differ from go-shellwords' on the raw string.
	"echo a\\\\\\\\$(date)",
	"'\\\\'$(x)",
	"echo a\\\\\\$(date)",
	"'\\'$$(x)",
	"echo ${X:-$}(1)",
}

// splitmix64, so the strings do not depend on math/rand's algorithms.
type rng uint64

func (r *rng) next() uint64 {
	*r += 0x9e3779b97f4a7c15
	z := uint64(*r)
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (r *rng) intn(n int) int { return int(r.next() % uint64(n)) }

// Every character that means something to go-shellwords, and some that do not.
var everything = []string{
	"'", "\"", "\\", "`", "$", "(", ")", ";", "&", "|", "<", ">", "#",
	" ", "\t", "\r", "\n", "\x00", "a", "b", "t", "n", "1", "2", "{", "}",
	"=", "é", "中", "😀", " ", " ",
}

// Quoting alone: no parentheses or operators, which end most strings early.
var quoting = []string{
	"'", "\"", "\\", "`", "$", "#", " ", " ", "\t", "\r", "\n", "\x00",
	"a", "b", "t", "n", "1", "é", "😀", " ",
}

// Pieces of commands.
var pieces = []string{
	"echo", "sh", "-c", "a", "b c", " ", "  ", "\t", "\n", "\r", "$(", ")",
	"(", "$", "${X}", "$X", "$$", "`", "'", "\"", "\\", "\\\\", "\\'", "\\\"",
	"\\`", "\\$", "\\t", "\\n", "\\ ", "2>", ">", ">&", "1", "12", "&&", "|",
	";", "<", "#", "#x", "''", "\"\"", "'a b'", "\"a b\"", "\"$(x)\"",
	"'\\'", "é", "\x00",
}

func random(r *rng, alphabet []string, most int) string {
	n := r.intn(most + 1)
	var b strings.Builder
	for range n {
		b.WriteString(alphabet[r.intn(len(alphabet))])
	}
	return b.String()
}

func main() {
	var inputs []string
	seen := map[string]bool{}
	add := func(s string) {
		if !seen[s] {
			seen[s] = true
			inputs = append(inputs, s)
		}
	}
	for _, c := range cases {
		add(c)
	}
	for _, path := range os.Args[1:] {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			var line struct {
				Text *string `json:"text"`
			}
			if err := json.Unmarshal(sc.Bytes(), &line); err != nil || line.Text == nil {
				fmt.Fprintf(os.Stderr, "%s: not a {\"text\": ...} line: %s\n", path, sc.Text())
				os.Exit(1)
			}
			add(*line.Text)
		}
		if err := sc.Err(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		f.Close()
	}
	r := rng(20261010)
	for range 2000 {
		add(random(&r, everything, 16))
	}
	for range 2000 {
		add(random(&r, quoting, 24))
	}
	for range 2000 {
		add(random(&r, pieces, 10))
	}

	out := bufio.NewWriter(os.Stdout)
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)
	for _, text := range inputs {
		words, err := shellwords.Parse(text)
		var line any
		if err != nil {
			line = struct {
				Text  string `json:"text"`
				Error string `json:"error"`
			}{text, err.Error()}
		} else {
			line = struct {
				Text  string   `json:"text"`
				Words []string `json:"words"`
			}{text, words}
		}
		if err := enc.Encode(line); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if err := out.Flush(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
