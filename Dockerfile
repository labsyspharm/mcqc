FROM python:3.12

COPY . src

RUN pip install --no-cache-dir build \
    && python -m build -w src \
    && pip install --no-cache-dir src/dist/mcqc*.whl \
    && rm -r src
