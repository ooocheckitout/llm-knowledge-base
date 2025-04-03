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

# Chroma Database

```shell
docker run -it --rm --name chromadb -p 8000:8000 -v ./src/.chroma:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE chromadb/chroma
```

#  

# Dozzle

```shell
docker run -it --rm --name dozzle -p 8080:8080 -v /run/user/1000/docker.sock:/var/run/docker.sock ghcr.io/amir20/dozzle:latest
```