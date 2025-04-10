import ast
import unittest
import generated_rules_R10  # Le module généré contenant la règle R10

class TestRuleR10(unittest.TestCase):
    def setUp(self):
        # Capture les messages générés par la fonction report
        self.messages = []
        generated_rules_R10.report = lambda msg: self.messages.append(msg)

    def run_rule(self, code):
        # Parse le code source et exécute la règle R10 sur l'AST
        ast_node = ast.parse(code)
        generated_rules_R10.rule_R10(ast_node)

    def test_detect_memory_not_freed_tensorflow(self):
        """
        Test d'un usage de TensorFlow sans clear_session() dans une boucle
        => On s'attend à au moins 1 message (car aucune API de libération mémoire n'est appelée)
        """
        code = """
import tensorflow as tf
for _ in range(100):
    model = tf.keras.Sequential([tf.keras.layers.Dense(10) for _ in range(10)])
"""
        self.run_rule(code)
        self.assertTrue(
            len(self.messages) > 0,
            "Un avertissement devrait être généré car aucune API de libération mémoire n'est appelée en TensorFlow"
        )

    def test_detect_memory_not_freed_pytorch(self):
        """
        Test d'un usage de PyTorch sans .detach() 
        => On s'attend à au moins 1 message (car .detach() n'est jamais appelé)
        """
        code = """
import torch

# Création explicite d'un tenseur via torch
tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
tensor2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Appel de matmul sur un tenseur défini par torch.
output = tensor1.matmul(tensor2)

# Appel de add sans utiliser .detach(), ce qui devrait être détecté comme problème par la règle R10.
result = output.add(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))

print(result)
"""
        self.run_rule(code)
        self.assertTrue(
            len(self.messages) > 0,
            "Un avertissement devrait être généré car .detach() n'est jamais appelé pour libérer la mémoire en PyTorch"
        )

    def test_detect_memory_not_freed_tensorflow_correct(self):
        """
        Test d'un usage correct de TensorFlow avec clear_session() dans une boucle
        => On ne devrait pas avoir de message
        """
        code = """
import tensorflow as tf
for _ in range(100):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([tf.keras.layers.Dense(10) for _ in range(10)])
"""
        self.run_rule(code)
        self.assertEqual(
            len(self.messages), 0,
            "Aucun avertissement ne devrait être généré, car clear_session() est appelé"
        )

    def test_detect_memory_not_freed_pytorch_correct(self):
        """
        Test d'un usage correct de PyTorch avec .detach() 
        => On ne devrait pas avoir de message
        """
        code = """
import torch
for _ in range(100):
    output = tensor1.matmul(tensor2)
    result = output.detach()
"""
        self.run_rule(code)
        self.assertEqual(
            len(self.messages), 0,
            "Aucun avertissement ne devrait être généré, car .detach() est appelé"
        )

    # Tests pour éviter les faux positifs

    def test_false_positive_git(self):
        """
        Test qu'un appel provenant de git (ex: repo.git.add) ne déclenche pas la règle.
        """
        code = """
import git
repo = git.Repo.init("dummy_repo")
repo.git.add(A=True)
"""
        self.run_rule(code)
        self.assertEqual(
            len(self.messages), 0,
            "Aucun avertissement ne devrait être généré pour un appel git.add"
        )

    def test_false_positive_os(self):
        """
        Test qu'un appel à une fonction os ne déclenche pas la règle.
        """
        code = """
import os
os.path.join("a", "b")
"""
        self.run_rule(code)
        self.assertEqual(
            len(self.messages), 0,
            "Aucun avertissement ne devrait être généré pour os.path.join"
        )

    def test_false_positive_numpy(self):
        """
        Test qu'un appel à numpy.add ne déclenche pas la règle (numpy n'est pas PyTorch).
        """
        code = """
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.add(a, b)
"""
        self.run_rule(code)
        self.assertEqual(
            len(self.messages), 0,
            "Aucun avertissement ne devrait être généré pour np.add"
        )

if __name__ == '__main__':
    unittest.main()
