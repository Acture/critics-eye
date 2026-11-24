FROM python:3.13-slim

WORKDIR /workspace

ENV PATH="/root/.local/bin:${PATH}"

# Same base packages as the MotifBench task image, plus the toolchain
# Harbor's MiniSweAgent.install() runs before the first trial:
# uv on Python 3.12 and mini-swe-agent==2.2.8 with the LiteLLM extras.
RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		ca-certificates curl git ripgrep build-essential \
	&& rm -rf /var/lib/apt/lists/* \
	&& pip install --no-cache-dir pypdf==7.1.1 \
	&& curl -LsSf https://astral.sh/uv/install.sh | sh \
	&& uv python install 3.12 \
	&& uv tool install --python 3.12 mini-swe-agent==2.2.8 \
		--with 'litellm[mcp]' --with orjson --with fastapi \
	&& mini-swe-agent --help >/dev/null
