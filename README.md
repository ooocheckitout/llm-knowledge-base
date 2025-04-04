# Build Docker Image

```shell
docker build -t ghcr.io/ooocheckitout/bitspire:latest --platform linux/amd64,linux/arm64 .
```

```shell
docker buildx create --use
docker buildx build -t ghcr.io/ooocheckitout/bitspire:latest --platform linux/amd64,linux/arm64 --push .

docker buildx build -t ghcr.io/ooocheckitout/bitspire:latest --platform linux/amd64 --load .
docker buildx build -t ghcr.io/ooocheckitout/bitspire:latest --platform linux/arm64 --load .
```

```shell
docker run -it --rm --env-file src/.env --name bitspire ghcr.io/ooocheckitout/bitspire:latest
```

# Push Docker Image

```shell
echo GITHUB_PAT_TOKEN | docker login ghcr.io -u solomoychenkoo@gmail.com --password-stdin

```

```shell
docker push ghcr.io/ooocheckitout/bitspire:latest
```