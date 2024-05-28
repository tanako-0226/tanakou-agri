# integrated-training-team-aguri
チーム「アグリ」のリポジトリ

チームアグリリポジトリのクローン
```
git clone https://github.com/ds-products/integrated-training-team-aguri.git
```
dockerの立ち上げ
```
docker compose up .
```
streamlitが起動するので、[ブラウザ](http://localhost:8501)で確認

コンテナへの入り方
```
docker-compose exec app bash
```
**git作業**

作業を始めるとき、、、

ローカルのmainブランチに移動
```
git switch main
```
GitHub の main ブランチから最新の情報を取得
```
git pull origin main
```
ローカルで新しい作業ブランチを切る
```
git checkout -b feature-branch
```

作業が終わったら、、、

作業ブランチで変更を加え、コミット
```
git add .
git commit -m "Add new feature"
```
リモートへプッシュ
```
git push origin main
```
