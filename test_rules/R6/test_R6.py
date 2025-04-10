import ast
import unittest
import generated_rules_R6  # Ce module doit contenir la fonction rule_R6

class TestGeneratedRulesR6(unittest.TestCase):
    def setUp(self):
        # Réinitialisation de la liste des messages pour chaque test
        self.messages = []
        def report(message):
            self.messages.append(message)
        # Monkey-patching de la fonction report du module généré
        generated_rules_R6.report = report

    def run_rule(self, code):
        ast_node = ast.parse(code)
        generated_rules_R6.rule_R6(ast_node)

    def test_detect_with_transformers_set_seed(self):
        """Test du cas où transformers.set_seed(X) est utilisé"""
        code = """
import transformers
transformers.set_seed(42)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0,
                         "Aucun message n'est attendu si transformers.set_seed est utilisé")

    def test_detect_with_random_seed(self):
        """Test du cas où random.seed(X) est utilisé"""
        code = """
import random
random.seed(42)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0,
                         "Aucun message n'est attendu si random.seed est utilisé")

    def test_detect_with_np_random_seed(self):
        """Test du cas où np.random.seed(X) est utilisé"""
        code = """
import numpy as np
np.random.seed(42)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0,
                         "Aucun message n'est attendu si np.random.seed est utilisé")

    def test_detect_with_torch_manual_seed(self):
        """Test du cas où torch.manual_seed(X) est utilisé"""
        code = """
import torch
torch.manual_seed(1)

# Exemple de code supplémentaire pour simuler un contexte de modèle
import torch.nn as nn
class TitanicSimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12, 12)
    def forward(self, x):
        return self.linear1(x)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0,
                         "Aucun message n'est attendu si torch.manual_seed est utilisé")

    def test_detect_with_torch_use_deterministic_algorithms(self):
        """Test du cas où torch.use_deterministic_algorithms(True) est utilisé"""
        code = """
import torch
torch.use_deterministic_algorithms(True)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0,
                         "Aucun message n'est attendu si torch.use_deterministic_algorithms(True) est utilisé")

    def test_detect_missing_deterministic_algorithm(self):
        """Test du cas où aucune méthode de reproductibilité n'est présente"""
        code = """
import torch
model = torch.nn.Linear(10, 2)
"""
        self.run_rule(code)
        self.assertGreater(len(self.messages), 0,
                           "Un message devrait être généré si aucune méthode de reproductibilité n'est utilisée")

if __name__ == '__main__':
    unittest.main()
