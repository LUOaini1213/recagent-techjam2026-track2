# Official KuaiRand-Pure starter (vendored)

Do not edit `evaluate.py`. Label is `long_view`. Primary = mean(GAUC, nDCG@5).

```powershell
cd kuairand-starter-kit
python baseline.py --data_dir ..\data\raw\KuaiRand-Pure\data --model pop
python baseline.py --data_dir ..\data\raw\KuaiRand-Pure\data --model fm
```

FM is the number to beat: valid primary 0.6016, test 0.5946.
