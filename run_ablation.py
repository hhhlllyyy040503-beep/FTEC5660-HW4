"""Controlled cumulative ablation. Run from the assignment repository.
python run_ablation.py --variant B0
python run_ablation.py --all
python run_ablation.py --report
Only this experiment driver reads gold labels to compute the report, after retrieval.
"""
from pathlib import Path
import argparse, ast, csv, json, re, subprocess, sys, hashlib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ablation_results'
VARIANTS = {
 'B0': 'Naive: get_text + recursive splitter + whitespace BM25; global top-5',
 'V1': 'B0 + table_to_text serialization (same recursive splitter)',
 'V2': 'V1 + section-aware prose / table-row chunks with contextual headers',
 'V3': 'V2 + normalized tokenization, stopwords and removal of entity boilerplate',
 'V4': 'V3 + parsed company/type/year filtering (metadata-aware year rules)',
 'V5': 'V4 + per-company slots for comparisons',
 'V6': 'V5 + financial metric expansion and table/row/scope/year score boosts',
 'V7': 'V6 + LLM evidence selection; empty selections fall back to lexical top-2',
 'V8': 'V7 + evidence-level abstention; final submitted implementation',
}

# All stages reuse the submitted metadata extractor. This is necessary even in B0.
PRELUDE = r'''
from langchain_text_splitters import RecursiveCharacterTextSplitter
_ABLATION_STAGE = STAGE_VALUE
_final_chunk = chunk_document
_final_parse = parse_query
_final_retrieve = retrieve
_original_tokens = _tokens

def chunk_document(file_name, html):
    structured = _final_chunk(file_name, html)
    if _ABLATION_STAGE >= 2:
        return structured
    meta = {k: structured[0].metadata[k] for k in
            ('source', 'company', 'fiscal_year', 'doc_type')}
    soup = BeautifulSoup(html, 'html.parser')
    if _ABLATION_STAGE == 1:
        for table in list(soup.find_all('table')):
            pre = soup.new_tag('pre')
            pre.string = table_to_text(str(table))
            table.replace_with(pre)
    # B0 deliberately uses raw BeautifulSoup get_text, no table serialization.
    text = soup.get_text(separator=' ', strip=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1450, chunk_overlap=150)
    return [Document(page_content=t, metadata=dict(meta)) for t in splitter.split_text(text)]

def parse_query(question):
    if _ABLATION_STAGE < 4:
        return {'companies': [], 'fiscal_year': None, 'doc_type': 'annual_report'}
    return _final_parse(question)

def _tokens(text):
    return text.lower().split() if _ABLATION_STAGE < 3 else _original_tokens(text)

_final_build = build_index

def build_index(chunks):
    if _ABLATION_STAGE >= 3:
        return _final_build(chunks)
    from rank_bm25 import BM25Okapi
    chunks = list(chunks)
    groups = {}
    for i, c in enumerate(chunks):
        groups.setdefault(_key(c.metadata['company']), []).append(i)
    return {'chunks': chunks, 'groups': groups,
            'bm25': BM25Okapi([c.page_content.lower().split() or ['empty'] for c in chunks])}
'''

# For stages V3 onward, use precisely the final ranking implementation with
# independently controlled features. V8 delegates to the unmodified final function.
def make_source(source, stage):
    marker = '# Provided code below. Do not change it.'
    pos = source.rfind('# =============================================================================', 0, source.index(marker))
    overrides = PRELUDE.replace('STAGE_VALUE', str(stage))
    if stage < 3:
        overrides += '''
def retrieve(index, question):
    scores = index['bm25'].get_scores(_tokens(question))
    order = sorted(range(len(index['chunks'])), key=lambda i: (-float(scores[i]), i))
    return [index['chunks'][i] for i in order[:5]]
'''
    elif stage < 8:
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'retrieve')
        lines = source.splitlines(True)
        body = ''.join(lines[fn.lineno - 1:fn.end_lineno])
        body = body.replace('def retrieve(', 'def _controlled_retrieve(', 1)
        body = body.replace("    parsed = parse_query(question)",
                            "    parsed = parse_query(question) if _ABLATION_STAGE >= 4 else dict(companies=list(index['groups']), fiscal_year=None, doc_type=None)")
        body = body.replace("        return []\n    q = question.lower()", "        return []\n    q = question.lower()")
        body = body.replace("            if meta['doc_type'] != parsed['doc_type']:",
                            "            if _ABLATION_STAGE >= 4 and meta['doc_type'] != parsed['doc_type']:")
        body = body.replace("            if terms and any(t in lower for t in terms):", "            if _ABLATION_STAGE >= 6 and terms and any(t in lower for t in terms):")
        body = body.replace("            if meta.get('kind') == 'table':", "            if _ABLATION_STAGE >= 6 and meta.get('kind') == 'table':")
        body = body.replace("            if year is not None and meta['fiscal_year'] == year:", "            if _ABLATION_STAGE >= 6 and year is not None and meta['fiscal_year'] == year:")
        body = body.replace("_tokens(query + ' ' + ' '.join(terms))", "_tokens(query + (' ' + ' '.join(terms) if _ABLATION_STAGE >= 6 else ''))")
        body = body.replace('    selected_groups = []', '    selected_groups = []\n    all_candidates = []')
        body = body.replace('        candidates.sort(reverse=True)', '''        candidates.sort(reverse=True)
        all_candidates.extend(candidates)
        if _ABLATION_STAGE < 7:
            selected_groups.append([i for _, i in candidates[:5]])
            continue''')
        body = body.replace('        if not chosen:\n            return []', '        if not chosen:\n            chosen = [i for _, i in shortlist[:2]]')
        body = body.replace('    # Interleave evidence so each comparison company gets a slot first.', '''    if _ABLATION_STAGE < 5:
        all_candidates.sort(reverse=True)
        return [index['chunks'][i] for _, i in all_candidates[:5]]
    # Interleave evidence so each comparison company gets a slot first.''')
        body = body.replace('    for depth in range(2):', '    for depth in range(2 if _ABLATION_STAGE >= 7 else 5):')
        overrides += '\n' + body + '\nretrieve = _controlled_retrieve\n'
    else:
        overrides += '\nretrieve = _final_retrieve\n'
    # Record only the actually returned evidence. Official runner remains unchanged.
    overrides += '''
_ablation_retrieve = retrieve

def retrieve(index, question):
    result = _ablation_retrieve(index, question)
    trace_path = Path(__file__).parent / 'evidence.jsonl'
    with trace_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'question': question, 'chunks': [
            {'text': c.page_content, 'metadata': c.metadata} for c in result
        ]}, ensure_ascii=False) + '\\n')
    return result
'''
    # Correct a literal escape in the generated trace writer.
    overrides = overrides.replace("+ '\\\\n')", "+ '\\n')")
    result = source[:pos] + '\n' + overrides + '\n' + source[pos:]
    ast.parse(result)
    assert result[result.index(marker):] == source[source.index(marker):]
    return result


def run_variant(name, data):
    stage = list(VARIANTS).index(name)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'metrics.json').exists():
        print(name, 'already complete; skipped. Delete its output folder to rerun.')
        return
    source = (ROOT / 'hw4.py').read_text(encoding='utf-8')
    variant = make_source(source, stage)
    path = folder / 'hw4_variant.py'
    path.write_text(variant, encoding='utf-8')
    (folder / 'evidence.jsonl').unlink(missing_ok=True)
    with (folder / 'run.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(path), '--data-folder', str(data),
                                 '--output', str(folder / 'results.csv')],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = proc.wait()
    if code:
        raise RuntimeError(f'{name} failed; see {folder / "run.log"}')
    logtext = (folder / 'run.log').read_text()
    if re.search(r'Generator failed|Traceback|failed on a table|WARNING:.*limit', logtext):
        raise RuntimeError(f'{name} contains errors; inspect its log before accepting results.')
    metrics = score_csv(folder, data)
    metrics['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(name, 'complete:', metrics)


def score_csv(folder, data):
    rows = list(csv.DictReader((folder / 'results.csv').open()))
    gold = json.loads((data / 'ground_truth.json').read_text())
    answerable = [r for r in rows if gold[r['id']]['evidence']]
    unanswerable = [r for r in rows if not gold[r['id']]['evidence']]
    truth = lambda x: str(x).lower() == 'true'
    mean = lambda xs: sum(xs) / len(xs) if xs else 0
    m = dict(accuracy=mean([truth(r['correct']) for r in rows]),
             recall=mean([truth(r['retrieval_hit']) for r in answerable]),
             kept=mean([bool(r['retrieved']) for r in answerable]),
             rejected=mean([not r['retrieved'] for r in unanswerable]),
             parse=mean([float(r['parse_score']) for r in rows]),
             errors=sum(not truth(r['correct']) for r in rows))
    log = (folder / 'run.log').read_text()
    usage = re.search(r'Your usage: (\{[^\n]+\})', log)
    generator = re.search(r'Generator usage: (\{[^\n]+\})', log)
    stats = [ast.literal_eval(x.group(1)) for x in (usage, generator) if x]
    m['llm_calls'] = sum(x['llm_calls'] for x in stats)
    m['cost_usd'] = sum(x['cost_usd'] for x in stats)
    tm = re.search(r'Time: ([\d.]+)s', log)
    m['seconds'] = float(tm.group(1)) if tm else None
    return m


def write_report():
    missing = [name for name in VARIANTS if not (OUT / name / 'metrics.json').exists()]
    if missing:
        raise RuntimeError('Run all variants before reporting: ' + ', '.join(missing))
    metrics = {n: json.loads((OUT / n / 'metrics.json').read_text()) for n in VARIANTS}
    hashes = {m.get('source_sha256') for m in metrics.values()}
    current = hashlib.sha256((ROOT / 'hw4.py').read_bytes()).hexdigest()
    if hashes != {current}:
        raise RuntimeError('Source changed between experiments. Rerun affected variants before reporting.')
    with (OUT / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'change', *next(iter(metrics.values())).keys()])
        writer.writeheader()
        for n,m in metrics.items():writer.writerow(dict(variant=n, change=VARIANTS[n], **m))
    report = '''## Homework 4 solution

### Pipeline

```mermaid
flowchart TD
    A[HTML reports] --> B[Parse metadata and serialize tables]
    B --> C[Section-aware prose and table-row chunks]
    C --> D[BM25 lexical index]
    Q[Question] --> E[Extract companies, requested year and document type]
    E --> F[Filter candidates using metadata and textual year coverage]
    D --> F
    F --> G[Rank per company: metric, table row and scope boosts]
    G --> H[LLM selects one or two evidence chunks per company]
    H --> I{Evidence available for every company?}
    I -->|Yes| J[Interleave at most five original chunks]
    I -->|No| K[Return empty list: NOT_FOUND]
    J --> L[Fixed provided generator]
```

HTML tables are serialized with each row label, column year and printed value together, including the caption and its units. The final parser creates separate table-row and prose chunks, with section/caption context, a maximum of 1,500 characters, and source, company, fiscal-year and document-type metadata. BM25 provides lexical retrieval without embeddings. The question parser extracts all requested companies, including absent companies; word boundaries prevent “operating” from being classified as a broker “rating”. Exact company and document-type filtering prevents similarly named companies from being confused. A passage may inherit its document year for prose; prior-year table data and forecasts can be retrieved through their explicitly printed years. Normalized tokens and generic financial-concept rules boost exact financial rows and the requested segment or region. Up to twelve lexical candidates per company are passed to the permitted model for evidence selection. Comparison evidence is selected separately for each company and interleaved, with at most five original chunks returned. An unknown company, absence of year/type candidates, or an empty evidence selection causes abstention. No generated text is inserted into chunks; the final answer is written by the unchanged provided generator.

### Experimental setup

All nine configurations use the same 40-document, 60-question public test set, the provided generator, the same model settings, and the same five-chunk limit. The generator and official scoring code are unchanged. Each variant is executed in a fresh Python process. B0 uses BeautifulSoup get_text, RecursiveCharacterTextSplitter (1,450 characters, overlap 150), required metadata, and rank_bm25 with lowercased whitespace tokens over the whole corpus. It does not parse questions and never abstains. Adjacent variants change exactly one configured component; a component's internal subrules are enabled together and are not interpreted as independently measured effects. No gold labels are used by indexing, query parsing, retrieval, or evidence selection. Gold data is read by the official scorer and this report generator only after predictions. Results below come from the CSV and logs produced by each configuration, with retained source hashes and evidence traces. Runs are not assumed to be bitwise deterministic, even at temperature zero.

### Cumulative ablation

| Variant | Single component changed from previous row | Recall@5 | Kept | Rejected | Answer accuracy | Seconds | API calls | Estimated USD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
    pct = lambda x:f'{100*x:.1f}%'
    for n,m in metrics.items():
        report += f"| {n} | {VARIANTS[n]} | {pct(m['recall'])} | {pct(m['kept'])} | {pct(m['rejected'])} | {pct(m['accuracy'])} | {m['seconds']} | {m['llm_calls']} | {m['cost_usd']:.4f} |\n"
    report += '''
Kept is the share of answerable questions returning evidence; Rejected is the share of unanswerable questions returning no chunks. These distinguish false abstention from correct rejection. The scored parse field is not meaningful for B0–V3 because query parsing is deliberately disabled there. Model-call counts and estimated costs include both retrieval-side calls and the fixed generator, using the assignment's price constants rather than billing statements. V4 introduces legitimate empty results through metadata filtering; V8 adds evidence-level rejection, rather than claiming to introduce all forms of abstention.

### Analysis of each change

'''
    explanations={
     'V1':'Table serialization preserves row/column/value associations while leaving the splitter and retrieval unchanged.',
     'V2':'Structural chunking replaces fixed splitting and adds section/caption context, reducing ambiguity around financial rows. These are one chunking-policy component, not separately measured interventions.',
     'V3':'The tokenization/query-normalization component removes boilerplate and entity terms and normalizes punctuation, keeping the scoring policy otherwise unchanged.',
     'V4':'The metadata-filtering component introduces query parsing and exact company/type/year coverage, preventing wrong-entity and unsupported-year retrieval. It can also return an empty result when no eligible document exists.',
     'V5':'Per-company slot allocation protects comparison questions from one company occupying all five positions.',
     'V6':'The financial-ranking component adds concept expansion, exact table-row boosts, scope matching and a same-year preference; their individual contributions are not separately identified.',
     'V7':'The LLM evidence selector reranks the twelve lexical candidates for each company. Empty selections still fall back to lexical evidence, isolating reranking from evidence-level abstention.',
     'V8':'Evidence-level abstention accepts an empty model selection instead of falling back to lexical candidates. It can improve rejection while risking false refusals on answerable questions.',
    }
    names=list(VARIANTS)
    for i,n in enumerate(names[1:],1):
        prev=metrics[names[i-1]];m=metrics[n]
        da=100*(m['accuracy']-prev['accuracy']);dr=100*(m['recall']-prev['recall'])
        verdict='improved' if da>0 else 'reduced' if da<0 else 'did not change'
        report += f"**{n}.** {explanations[n]} Observed answer accuracy {verdict} ({da:+.1f} percentage points); recall changed by {dr:+.1f} points, kept by {100*(m['kept']-prev['kept']):+.1f}, and rejected by {100*(m['rejected']-prev['rejected']):+.1f}. "
        report += ('This run supports a benefit for this component in the preceding configuration. ' if da>0 else 'This run does not establish an accuracy benefit for this component; an unchanged or negative result is retained rather than omitted. ')
        report += 'These are conditional, single-run public-set effects, not proof of independent causal gains or private-set performance.\n\n'
    report+='### Error analysis\n\n'
    previous=None
    for name in VARIANTS:
        rows=list(csv.DictReader((OUT/name/'results.csv').open()))
        wrong=[r for r in rows if r['correct'].lower()!='true']
        gold=json.loads((ROOT/'public_test'/'ground_truth.json').read_text())
        false_reject=[r for r in wrong if gold[r['id']]['evidence'] and not r['retrieved']]
        miss=[r for r in wrong if gold[r['id']]['evidence'] and r['retrieved'] and r['retrieval_hit'].lower()!='true']
        evidence_wrong=[r for r in wrong if gold[r['id']]['evidence'] and r['retrieval_hit'].lower()=='true']
        unsupported=[r for r in wrong if not gold[r['id']]['evidence']]
        report+=f"- **{name}**: {len(wrong)} wrong; {len(false_reject)} false abstentions; {len(miss)} errors with missing evidence; {len(evidence_wrong)} wrong answers despite complete evidence; {len(unsupported)} failures on unanswerable questions.\n"
        if previous is not None:
            old={r['id']:r for r in previous}
            fixed=[r['id'] for r in rows if r['correct'].lower()=='true' and old[r['id']]['correct'].lower()!='true']
            reg=[r['id'] for r in rows if r['correct'].lower()!='true' and old[r['id']]['correct'].lower()=='true']
            report+=f"  Fixed from preceding variant: {', '.join(fixed) or 'none'}. Regressions: {', '.join(reg) or 'none'}.\n"
        previous=rows
    finalrows=list(csv.DictReader((OUT/'V8'/'results.csv').open()))
    wrong=[r for r in finalrows if r['correct'].lower()!='true']
    if wrong:
        report+='\nFinal-variant errors for inspection:\n\n| ID | Question | Answer | Expected |\n|---|---|---|---|\n'
        for r in wrong:
            report+='| '+' | '.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ('id','question','answer','expected'))+' |\n'
    else:
        report+='\nThe final variant has no observed errors on these 60 public questions. This does not rule out errors on new companies, differently structured HTML, ambiguous language, or hidden questions.\n'
    report+='''
### Limitations and reproducibility

The public set was also used during development, so these results are not an unbiased held-out estimate. Multiple ranking rules are grouped within one financial-ranking component; finer ablations would be needed to attribute gains to each rule. The model can vary between runs, and API failures cause deterministic lexical fallbacks; logs must be checked before accepting a run. Candidate selection may still miss evidence outside the top twelve, and aggressive rejection may lose answerable questions. `run_ablation.py` contains the precise configuration switches; `ablation_results/<variant>/` retains the executed code, per-question CSV, returned-evidence trace, log and metrics. Reproduce using `python run_ablation.py --all` and `python run_ablation.py --report` with the same final hw4.py. Never commit API keys, .env files or prebuilt index files.
'''
    target=ROOT/'README.md'
    existing=target.read_text() if target.exists() else '# FTEC5660 Homework 4\n\n'
    # Replace only the dedicated Task 5 section; preserve any following Task 6 section.
    start=re.search(r'^## Homework 4 solution\s*$',existing,re.M)
    if start:
        tail=existing[start.end():]
        next_heading=re.search(r'^## (?!#).+',tail,re.M)
        end=start.end()+next_heading.start() if next_heading else len(existing)
        existing=existing[:start.start()]+report+'\n'+existing[end:]
    else:existing+='\n'+report
    target.write_text(existing)
    print('Generated README.md and ablation_results/summary.csv from real runs.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--data-folder',default='public_test')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.report:write_report();return
    data=Path(args.data_folder).resolve()
    if not (data/'ground_truth.json').exists():raise ValueError('Public ground_truth.json required for analysis.')
    if not args.all and not args.variant:parser.error('Use --variant, --all or --report')
    for n in VARIANTS if args.all else [args.variant]:run_variant(n,data)

if __name__=='__main__':main()
