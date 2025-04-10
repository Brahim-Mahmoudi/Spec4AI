#!/usr/bin/env bash

# Boucle principale pour R1 à R22
for X in {1..22} "11bis"
#for X in "11bis"
do
  echo "=== Traitement de R$X ==="

  dsl_file="test_rules/R${X}/test_R${X}.dsl"
  output_file="test_rules/R${X}/generated_rules_${X}.py"
  test_file="test_rules/R${X}/test_R${X}.py"

  if [ ! -f "$dsl_file" ]; then
      echo "Fichier DSL $dsl_file non trouvé, passage à R$X suivant."
      continue
  fi

  echo "=== Génération des règles pour R$X ==="
  python parse_dsl.py --dsl "$dsl_file" --output "$output_file"

  timeout=20
  start_time=$(date +%s)
  timedout=false
  while [ ! -f "$output_file" ]; do
      sleep 0.1
      current_time=$(date +%s)
      if [ $((current_time - start_time)) -ge $timeout ]; then
          echo "Timeout: Le fichier $output_file n'a pas été créé après $timeout secondes. Passage à R$X suivant."
          timedout=true
          break
      fi
  done

  if [ "$timedout" = true ]; then
      continue
  fi

  if [ ! -f "$test_file" ]; then
      echo "Fichier de test $test_file non trouvé, passage à R$X suivant."
      continue
  fi

  echo "=== Exécution des tests pour R$X ==="
  pytest "$test_file"
done
