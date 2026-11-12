import pandas as pd
import pytest
from behaviour.scorers import batch

def test_manifest_window_alignment_and_duplicate_runs(tmp_path):
    config = '[video]\npx_per_m = 1\n[window]\nstart_s = 0\nduration_s = 180\n'
    (tmp_path / 'a.toml').write_text(config)
    (tmp_path / 'b.toml').write_text(config.replace('start_s = 0', 'start_s = 10'))
    (tmp_path / 'events.csv').write_text('video_id,behaviour,start,stop,event_type\nv,grooming,20,30,state\n')
    row = dict(run_id='manual', tool='boris', video_id='v', config='a.toml',
               export='events.csv', format='intervals', behaviours='grooming')
    candidate = dict(row, run_id='candidate', tool='labgym', config='b.toml')
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame([row,candidate]).to_csv(manifest,index=False)
    with pytest.raises(ValueError,match='Mismatched observation windows'):
        batch(manifest,tmp_path/'bad',None)
    assert not (tmp_path/'bad').exists()
    pd.DataFrame([row,row]).to_csv(manifest,index=False)
    with pytest.raises(ValueError,match='must appear once'):
        batch(manifest,tmp_path/'duplicate',None)
    candidate['config']='a.toml'
    pd.DataFrame([row,candidate]).to_csv(manifest,index=False)
    batch(manifest,tmp_path/'good','manual')
    metrics=pd.read_csv(tmp_path/'good/metrics.csv')
    durations=metrics[metrics.metric=='grooming_time_s'].value
    assert durations.tolist()==[10,10]
    assert (tmp_path/'good/run.json').exists()
