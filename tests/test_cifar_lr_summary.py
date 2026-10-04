import json
import pytest
from benchmarks.summarize_cifar_lr_screen import summarize


def test_incomplete_screen_cannot_be_ranked(tmp_path):
    (tmp_path/'screen.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='completed'):summarize(tmp_path)


def test_repeat_mismatch_cannot_be_ranked(tmp_path):
    (tmp_path/'screen.json').write_text(json.dumps({'status':'repeat_mismatch'}))
    with pytest.raises(ValueError,match='reproducible'):summarize(tmp_path)


@pytest.mark.parametrize('multiple_controls',[False,True])
@pytest.mark.parametrize('rates',[(.0005,.001,.002),(.000125,.00025,.0005)])
def test_summary_excludes_repeat_from_selection_and_preserves_missing_wall(tmp_path,multiple_controls,rates):
    runs=[]
    for policy in ['qr10','warm10','qr20']:
        for rate,ce in zip(rates,[2.,1.5,1.7]):
            observation=dict(step=3500,validation=dict(ce=ce,accuracy=.6),
                             training_seconds=80.,maximum_gram_error=.001,reasons=['fixed_updates'])
            runs.append(dict(label=f'{policy}_lr{rate:g}',policy=policy,matrix_lr=rate,
                             optimizer_ms=12.,observations=[observation]))
    # A repeat is a reproducibility control, never a fourth LR candidate.
    original=runs[1]['label']
    repeat={**runs[1],'label':original+'_repeat',
            'observations':[{**runs[1]['observations'][0],'validation':dict(ce=.1,accuracy=.9)}]}
    pairs=[[original,repeat['label']]]
    if multiple_controls:
        repeat.update(label=original+'_repeat_mid',repeat_of=original)
        end={**repeat,'label':original+'_repeat_end'}
        runs.append(end);pairs=[[original,repeat['label']],[original,end['label']]]
    runs.append(repeat)
    report=dict(status='completed',config={'lr_screen':{'equal_updates':3500,'wall_seconds':75.,'control_pairs':pairs,
                'bracket':'lower' if multiple_controls else 'original'}},
                runs=runs,control_repeat={'model_hash_equal':True},paired_fixed_update_checks_passed=True,
                all_in_seconds=900.,gpu_uuid='test')
    (tmp_path/'screen.json').write_text(json.dumps(report))
    result=summarize(tmp_path)
    assert all(x['matrix_lr']==rates[1] and x['validation_ce']==1.5 and not x['boundary_winner']
               for x in result['selected_by_equal_update_ce'])
    assert all(x['wall_ce'] is None and x['wall_overshoot_seconds'] is None for x in result['rows'])

    assert len(result['control_timing_repeats'])==(2 if multiple_controls else 1)

    if multiple_controls:
        assert result['timing_interpretation_gate']['passed']
        report['runs'][-1]['observations'][0]['training_seconds']=100.
        (tmp_path/'screen.json').write_text(json.dumps(report))
        assert not summarize(tmp_path)['timing_interpretation_gate']['passed']
