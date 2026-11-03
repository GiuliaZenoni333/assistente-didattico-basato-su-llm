"""Confronto dei quattro esperimenti RAG """
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'evaluation'
OUT = EVAL / 'comparison'
SCORES = ('correctness', 'completeness', 'source_support')
EXPERIMENTS = {
    'llama_progr1': EVAL / 'llama_3.2' / 'progr1',
    'phi_progr1': EVAL / 'phi4' / 'progr1',
    'llama_sas': EVAL / 'llama_3.2' / 'sas',
    'phi_sas': EVAL / 'phi4' / 'sas',
}

SERIES_STYLE = {
    'llama_progr1': {
        'label': 'Llama 3.2 - Programmazione I',
        'color': '#9AA7B3',
        'hatch': None
    },
    'phi_progr1': {
        'label': 'Phi-4-mini - Programmazione I',
        'color': '#123A6A',
        'hatch': None
    },
    'llama_sas': {
        'label': 'Llama 3.2 - SAS',
        'color': '#EAF2F9',
        'hatch': None
    },
    'phi_sas': {
        'label': 'Phi-4-mini - SAS',
        'color': '#2D6DA3',
        'hatch': None
    },
}


def read_jsonl(path):
    records = {}
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}, riga {line_number}: {exc}') from exc
            if record['id'] in records:
                raise ValueError(f'ID duplicato {record["id"]} in {path}')
            records[record['id']] = record
    return records


def manual_file(folder):
    candidates = sorted((folder / 'manual').glob('*final*.jsonl'))
    if len(candidates) != 1:
        raise ValueError(f'Atteso un file manuale finale in {folder / "manual"}; trovati: {candidates}')
    return candidates[0]


def metrics_file(folder):
    candidates = []
    for path in sorted((folder / 'metrics').glob('*.json')):
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        retrieval = obj.get('retrieval')
        if isinstance(retrieval, dict) and all(
            key in retrieval for key in ('hit_at_1', 'hit_at_3', 'hit_at_5', 'hit_at_8', 'mrr')
        ):
            candidates.append((path, obj))
    if len(candidates) != 1:
        raise ValueError(
            f'Atteso un solo JSON delle metriche con retrieval in {folder / "metrics"}; '
            f'trovati: {[str(path) for path, _ in candidates]}'
        )
    return candidates[0]


def retrieval_scores(obj):
    source = obj['retrieval']
    output = {'retrieval_evaluated': source['evaluated_questions']}
    for k in (1, 3, 5, 8):
        raw = source[f'hit_at_{k}']
        output[f'hit_at_{k}'] = round(raw['percentage'] / 100, 4) if isinstance(raw, dict) else float(raw)
    output['mrr'] = float(source['mrr'])
    return output


def is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_error(value):
    return is_numeric(value) and value < 1


def average(records, field):
    values = [
        r[field] for r in records
        if is_numeric(r[field])
    ]
    return round(sum(values) / len(values), 4) if values else None


def is_problematic(r):
    unsupported = (
        is_error(r['source_support'])
        and r['abstention'] != 1
    )
    return (
        is_error(r['correctness'])
        or is_error(r['completeness'])
        or unsupported
        or r['abstention'] == 0
    )


def summarize(manual, metrics):
    records = list(manual.values())

    if len(records) != metrics["total_questions"]:
        raise ValueError(
            f'Numero di valutazioni non coerente: '
            f'manual={len(records)}, '
            f'metrics={metrics["total_questions"]}'
        )
    for r in records:
        for k in SCORES:
            allowed = (0, 0.5, 1, 'N/A')
            if isinstance(r[k], bool) or r[k] not in allowed:
                raise ValueError(f'Punteggio non valido, ID {r["id"]}, {k}={r[k]}')
        if isinstance(r['abstention'], bool) or r['abstention'] not in (0, 1, 'N/A'):
            raise ValueError(f'Astensione non valida, ID {r["id"]}')

    ans = [r['abstention'] for r in records if r['abstention'] != 'N/A']
    summary = {k: average(records, k) for k in SCORES}
    summary.update({
        'total_questions': len(records),
        'correctness_evaluated': sum(is_numeric(r['correctness']) for r in records),
        'completeness_evaluated': sum(is_numeric(r['completeness']) for r in records),
        'source_support_evaluated': sum(is_numeric(r['source_support']) for r in records),
        'abstention_accuracy': round(sum(ans) / len(ans), 4) if ans else None,
        'abstention_evaluated': len(ans),
        'problematic_questions': sum(is_problematic(r) for r in records),
        'error_categories': {
            'incorrect_answer': sum(is_error(r['correctness']) for r in records),
            'incomplete_answer': sum(is_error(r['completeness']) for r in records),
            'unsupported_answer': sum(
                is_error(r['source_support']) and r['abstention'] != 1 for r in records
            ),
            'incorrect_abstention': sum(r['abstention'] == 0 for r in records),
        },
    })

    summary.update(retrieval_scores(metrics))
    for group_name in ('by_difficulty', 'by_question_type'):
        summary[group_name] = {}
        for label, group in metrics.get(group_name, {}).items():
            summary[group_name][label] = {
                'n': group.get('total'),
                **{
                    score: group[score].get('mean') if isinstance(group.get(score), dict) else None
                    for score in SCORES
                },
            }
    return summary


def compare_questions(llama, phi):
    if set(llama) != set(phi):
        raise ValueError('ID di Llama e Phi differenti nello stesso corso')
    output = {}
    for metric in SCORES:
        phi_better, llama_better, equal, not_comparable = [], [], [], []
        for i in sorted(llama):
            llama_value = llama[i][metric]
            phi_value = phi[i][metric]
            if llama_value == 'N/A' or phi_value == 'N/A':
                not_comparable.append(i)
            elif phi_value > llama_value:
                phi_better.append(i)
            elif phi_value < llama_value:
                llama_better.append(i)
            else:
                equal.append(i)
        output[metric] = {
            'phi_better_ids': phi_better,
            'llama_better_ids': llama_better,
            'equal_ids': equal,
            'not_comparable_ids': not_comparable,
        }
    return output



def plot_figures(summaries):
    import matplotlib.pyplot as plt
    import numpy as np

    figures = OUT / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    charts = {
        'answer_quality': (
            SCORES,
            'Qualità delle risposte',
            ['Correctness', 'Completeness', 'Source Support']
        ),
        'retrieval_comparison': (
            ('hit_at_1', 'hit_at_3', 'hit_at_5', 'hit_at_8', 'mrr'),
            'Prestazioni del retrieval',
            ['Hit@1', 'Hit@3', 'Hit@5', 'Hit@8', 'MRR']
        )
    }

    course_groups = {
        'Programmazione I': ('llama_progr1', 'phi_progr1'),
        'SAS': ('llama_sas', 'phi_sas')
    }

    for filename, (metrics, title, labels) in charts.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
        fig.patch.set_facecolor('white')

        for ax, (course_name, exp_names) in zip(axes, course_groups.items()):
            x = np.arange(len(metrics))
            width = 0.32

            for j, name in enumerate(exp_names):
                style = SERIES_STYLE[name]
                values = [summaries[name][m] for m in metrics]

                if any(v is None for v in values):
                    raise ValueError(f'Metrica mancante: {name}')

                ax.bar(
                    x + (j - 0.5) * width,
                    values,
                    width=width * 0.92,
                    label=style['label'],
                    color=style['color'],
                    hatch=style['hatch'],
                    edgecolor='#718096',
                    linewidth=0.7,
                    zorder=3
                )

            ax.set_title(course_name, fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', linestyle='--', alpha=0.25)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[0].set_ylabel('Punteggio medio (0–1)', fontsize=11)
        fig.suptitle(title, fontsize=15, fontweight='bold')

        handles, legend_labels = [], []
        from matplotlib.patches import Patch

        for name in EXPERIMENTS:
            style = SERIES_STYLE[name]
            handles.append(Patch(
                facecolor=style['color'],
                edgecolor='#718096',
                label=style['label']
            ))
            legend_labels.append(style['label'])

        fig.legend(
            handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.06),
            ncol=2,
            frameon=False,
            fontsize=9
        )

        fig.tight_layout(rect=[0, 0.09, 1, 0.95])

        fig.savefig(
            figures / f'{filename}.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        fig.savefig(
            figures / f'{filename}.pdf',
            bbox_inches='tight',
            facecolor='white'
        )

        plt.close(fig)
        print('Grafico:', figures / f'{filename}.png')


        
def main():
    summaries, manuals = {}, {}
    for name, folder in EXPERIMENTS.items():
        mpath = manual_file(folder)
        kpath, metrics = metrics_file(folder)
        manual = read_jsonl(mpath)
        summaries[name] = summarize(manual, metrics)
        manuals[name] = manual
        print(f'{name}: {mpath.name} + {kpath.name} | Hit@1={summaries[name]["hit_at_1"]}')
        print('  Quality denominators:', ', '.join(
            f'{key}={summaries[name][key + "_evaluated"]}/{len(manual)}'
            for key in SCORES
        ))

    paired = {
        course: compare_questions(manuals[f'llama_{course}'], manuals[f'phi_{course}'])
        for course in ('progr1', 'sas')
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'comparison_results.json').write_text(
        json.dumps({'experiments': summaries, 'question_comparison': paired}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    fields = [
        'experiment', 'total_questions', 'problematic_questions', *SCORES,
        'correctness_evaluated', 'completeness_evaluated',
        'source_support_evaluated', 'abstention_accuracy', 'abstention_evaluated',
        'retrieval_evaluated', 'hit_at_1', 'hit_at_3', 'hit_at_5', 'hit_at_8', 'mrr',
    ]
    with (OUT / 'comparison_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, summary in summaries.items():
            writer.writerow({'experiment': name, **{k: summary.get(k) for k in fields[1:]}})
    plot_figures(summaries)
    print('\nConfronto completato:', OUT / 'comparison_results.json', 'e', OUT / 'comparison_summary.csv')


if __name__ == '__main__':
    main()
