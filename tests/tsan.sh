#!/bin/sh
# Builds the queue and pool tests alone (no CUDA) under ThreadSanitizer and runs them.
# Run from the repo root on the host after build-trt is configured (it supplies the googletest sources).
# seccomp=unconfined lets setarch turn off ASLR, which g++-11's TSan cannot map around on this kernel.
set -e
G=build-trt/_deps/googletest-src/googletest
docker run --rm --security-opt seccomp=unconfined -u "$(id -u):$(id -g)" -v "$PWD":/work -w /work inference-trt sh -c "
  g++-11 -std=c++20 -fsanitize=thread -O1 -g -pthread -Isrc -I$G/include -I$G \
    tests/test_bounded_queue.cpp tests/test_pool.cpp $G/src/gtest-all.cc $G/src/gtest_main.cc -o /tmp/tsan_tests &&
  setarch -R /tmp/tsan_tests"
