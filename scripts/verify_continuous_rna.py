"""Audit all selections, normalizers and regenerate every expression prediction."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr
from dnabert_fno.data import sha256_file
from dnabert_fno.study import stable_hash,atomic_json
import continuous_rna_models as models
from run_continuous_rna import install_cache


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def independent_metrics(y,p):
    r=pearsonr(y,p,axis=0).statistic;r=np.nan_to_num(r,nan=0.)
    return float(r.mean()),float(np.mean(np.square(y-p))),r


def main():
    torch.set_num_threads(4);root=Path('runs/continuous_rna');protocol=read(root/'protocol.json');cfg=protocol['config'];data=Path(cfg['data_dir'])
    status=read(root/'status.json');assert status['phase']=='complete' and stable_hash(protocol)==status['study_id']
    for p,digest in protocol['code_sha256'].items():assert sha256_file(p)==digest
    assert sha256_file('docs/CONTINUOUS_RNA_PROTOCOL.md')==protocol['protocol_document_sha256']
    assert sha256_file(data/'source.json')==protocol['source_sha256']
    assert sha256_file(root/'features/manifest.json')==protocol['cache_manifest_sha256']
    source=read(data/'source.json')
    for name,digest in source['files_sha256'].items():assert sha256_file(data/name)==digest
    cache=read(root/'features/manifest.json');assert cache['encoder_hash_before']==cache['encoder_hash_after']
    assert sha256_file(root/'features/grid.npy')==cache['sha256']
    data_audit=read(root/'data_audit.json');assert data_audit['passed'] and data_audit['chunk_cross_split_aliases']==0
    assert data_audit['source_sha256']==sha256_file(data/'source.json') and data_audit['audit_script_sha256']==sha256_file('scripts/audit_continuous_rna.py')
    assert data_audit['original_label_and_coordinate_mapping_verified']
    for p,digest in source['sequence_response_sha256'].items():assert sha256_file(p)==digest
    rows=read(data/'rows.json');ids={s:np.array([r['global_id'] for r in rows if r['split']==s]) for s in ['train','dev','test']}
    labels=np.load(data/'labels.npy');norm=np.load(root/'normalizers.npz')
    np.testing.assert_array_equal(norm['label_mean'],labels[ids['train']].mean(0));np.testing.assert_array_equal(norm['label_sd'],labels[ids['train']].std(0))
    y=(labels-norm['label_mean'])/norm['label_sd'];install_cache(root/'features/grid.npy',cfg['device'])
    mmap=np.load(root/'features/grid.npy',mmap_mode='r');total=np.zeros(768,dtype=np.float64)
    for first in range(0,len(ids['train']),8):total+=mmap[ids['train'][first:first+8]].sum((0,1),dtype=np.float64)
    expected=(total/(len(ids['train'])*256)).astype(np.float32);np.testing.assert_allclose(expected,norm['feature_mean'],atol=1e-6,rtol=0)
    models.MASK_MEAN=torch.as_tensor(norm['feature_mean'],device=cfg['device'])
    training=read(root/'training_complete.json');summary=read(root/'summary.json')
    assert not training['test_used'] and len(training['all_training'])==36 and len(training['final'])==27 and len(summary['results'])==126
    assert sha256_file(root/'normalizers.npz')==training['normalizers_sha256']
    assert summary['test_gene_ids']==ids['test'].tolist()
    assert abs(summary['train_mean_baseline_mse']-float(np.mean(y[ids['test']]**2)))<1e-12
    maxdev=0.;maxtest=0.
    for number,r in enumerate(training['all_training'],1):
        dest=Path(r['path']);s=r['settings'];history=read(dest/'epochs.json');best=-float('inf');best_epoch=None
        for h in history:
            assert np.isfinite(h['train_mse'])
            if h['dev_macro_pearson']>best+cfg['min_delta']:best=h['dev_macro_pearson'];best_epoch=h['epoch']
        assert best==r['best_dev_macro_pearson'] and best_epoch==r['best_epoch'] and not r['test_evaluated']
        assert sha256_file(dest/'best.pt')==r['checkpoint_sha256'] and sha256_file(dest/'dev_predictions.npy')==r['dev_predictions_sha256']
        model=models.make_model(s['model'],cfg,s['seed']);assert models.head_hash(model)==r['head_initial_sha256']
        assert sum(p.numel() for p in model.parameters())==r['params']==models.count_params(s['model'],models.width_for(s['model']))
        cp=torch.load(dest/'best.pt',map_location=cfg['device'],weights_only=True);assert cp['epoch']==best_epoch and cp['settings']==s;model.load_state_dict(cp['state_dict'])
        p=models.predict(model,ids['dev'],s['bp'],cfg);old=np.load(dest/'dev_predictions.npy');maxdev=max(maxdev,float(np.max(np.abs(p-old))))
        np.testing.assert_allclose(p,old,atol=1e-6,rtol=0)
        macro,_,_=independent_metrics(y[ids['dev']],old);assert abs(macro-best)<1e-12
        del model
        atomic_json(root/'verification_progress.json',{'phase':'dev_regeneration','completed':number,'total':36})
    for key,entry in training['selection'].items():
        chosen=sorted(entry['candidates'],key=lambda r:(-r['best_dev_macro_pearson'],r['settings']['lr']))[0]
        assert chosen in training['final'] and chosen['settings']['lr']==entry['selected_lr']
        kind,bp=key.rsplit('_',1)
        final=[r for r in training['final'] if r['settings']['model']==kind and r['settings']['bp']==int(bp)]
        assert {r['settings']['seed'] for r in final}=={42,43,44} and all(r['settings']['lr']==entry['selected_lr'] for r in final)
    for seed in cfg['seeds']:assert len({r['head_initial_sha256'] for r in training['final'] if r['settings']['seed']==seed})==1
    expected={(kind,bp,seed,'intact',None) for kind in cfg['variants'] for bp in cfg['lengths_bp'] for seed in cfg['seeds']}
    expected|={(kind,65536,seed,'mask',None) for kind in cfg['variants'] for seed in cfg['seeds']}
    expected|={(kind,65536,seed,condition,ps) for kind in cfg['variants'] for seed in cfg['seeds'] for condition in ['shuffle','swap'] for ps in cfg['perturbation_seeds']}
    observed={(r['settings']['model'],r['settings']['bp'],r['settings']['seed'],r['condition'],r['perturbation_seed']) for r in summary['results']}
    assert observed==expected and len(observed)==126
    for number,r in enumerate(summary['results'],1):
        s=r['settings'];assert r['training'] in training['final'];path=Path(r['prediction_path']);assert sha256_file(path)==r['prediction_sha256']
        old=np.load(path);assert old.shape==(982,218) and np.isfinite(old).all()
        model=models.make_model(s['model'],cfg,s['seed']);model.load_state_dict(torch.load(Path(r['training']['path'])/'best.pt',map_location=cfg['device'],weights_only=True)['state_dict'])
        p=models.predict(model,ids['test'],s['bp'],cfg,r['condition'],r['perturbation_seed']);maxtest=max(maxtest,float(np.max(np.abs(p-old))))
        np.testing.assert_allclose(p,old,atol=1e-6,rtol=0);macro,mse,tracks=independent_metrics(y[ids['test']],old)
        assert abs(macro-r['metrics']['macro_pearson'])<1e-12 and abs(mse-r['metrics']['mse'])<1e-12
        np.testing.assert_allclose(tracks,r['metrics']['per_track_pearson'],atol=1e-12,rtol=0);del model
        atomic_json(root/'verification_progress.json',{'phase':'test_regeneration','completed':number,'total':126})
        if number%18==0:print(f'Verified test {number}/126',flush=True)
    result={'passed':True,'study_id':status['study_id'],'verification_script_sha256':sha256_file(__file__),'training_trials':36,'dev_regenerations':36,'test_regenerations':126,
            'max_dev_abs_difference':maxdev,'max_test_abs_difference':maxtest,'checks':['Source/code/cache/checkpoint hashes','Train-only label and feature normalizers','All dev-only checkpoint and LR choices',
            'Shared head initializations and parameter budgets','Exact evaluation-condition coverage','Independent SciPy Pearson and MSE recomputation','Every dev and test prediction regenerated']}
    atomic_json(root/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
