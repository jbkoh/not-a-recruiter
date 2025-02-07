
from typing import Dict
from tqdm import tqdm
from pdb import set_trace as bp
from typing import Optional

from dotenv import load_dotenv
import pandas as pd
import json
from pathlib import Path
from typer import Typer, Option

from .pipelines import ResumeScreener


app = Typer()


def format_output(all_results, requirements: Dict[str, str]):
    columns = [
        'filename',
        'name',
        'decision',
        'reason',
    ] + list(requirements.keys()) + ['error', 'error_detail', 'token_num']

    rows = []
    for filename, res in all_results.items():
        row = {
            'filename': filename,
            'decision': None,
            'reason': None,
            'name': None,
            'error': None,
            'error_detail': None,
            'token_num': res['llm']['replies'][0]._meta['usage']['total_tokens']
        }
        raw_text = res['llm']['replies'][0].text
        try:
            resp = json.loads(raw_text)
            resp['applicant_name']
            resp['decision']
            resp['reason']
        except Exception as e:
            aaa = e
            row['error'] = 'Could not parse response'
            row['error_detail'] = raw_text
            continue
        row['decision'] = resp['decision']
        row['name'] = resp['applicant_name']
        row['reason'] = resp['reason']
        for key in requirements.keys():
            row[key] = resp.get(key, None)
        rows.append(row)
    df = pd.DataFrame(rows)[columns]
    return df



@app.command()
def screen_multiple(
    resumes_dir: Path=Option(None),
    jd: Path=Option(None),
    config: Path=Option(None),
):
    if config:
        with open(config) as f:
            config = json.load(f)
        jd = Path(config['job_description'])
        resumes_dir = Path(config['resumes_dir'])

    screener = ResumeScreener(
        jd_file=jd,
        requirements=config['requirements'],
        additional_context=config['additional_context']
    )
    all_res = {}
    for i, resume in tqdm(enumerate(resumes_dir.iterdir())):
        res = screener.run(resume_file=resume)
        all_res[resume] = res
    df = format_output(all_res, requirements=config['requirements'])
    df.to_csv('output.csv', index=False)


@app.callback()
def setup(
    dotenv: Optional[str] = Option(".env", envvar="MODELCI_DOTENV"),
):
    load_dotenv(dotenv)


if __name__ == "__main__":
    app()


