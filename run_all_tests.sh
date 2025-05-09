#!/usr/bin/env bash

# Boucle principale pour R1 à R22
for X in {1..24} "11bis"
#for X in "11bis"
do
  echo "=== Analyze of R$X ==="

  dsl_file="test_rules/R${X}/test_R${X}.dsl"
  output_file="test_rules/R${X}/generated_rules_R${X}.py"
  test_file="test_rules/R${X}/test_R${X}.py"

  if [ ! -f "$dsl_file" ]; then
      echo "File DSL $dsl_file not find, go to next R$X."
      continue
  fi

  echo "=== Generation of rules for R$X ==="
  python parse_dsl.py --dsl "$dsl_file" --output "$output_file"

  timeout=20
  start_time=$(date +%s)
  timedout=false
  while [ ! -f "$output_file" ]; do
      sleep 0.1
      current_time=$(date +%s)
      if [ $((current_time - start_time)) -ge $timeout ]; then
          echo "Timeout: File $output_file couldn't be load after $timeout seconds. Go to next R$X."
          timedout=true
          break
      fi
  done

  if [ "$timedout" = true ]; then
      continue
  fi

  if [ ! -f "$test_file" ]; then
      echo "Test file $test_file not found, Go to next R$X."
      continue
  fi

  echo "=== Test Execution for R$X ==="
  pytest "$test_file"
done
