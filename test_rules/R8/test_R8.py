import ast
import unittest
import generated_rules_R8  # Le module généré contenant la règle R8

class TestPyTorchCallMethodMisuse(unittest.TestCase):
    def setUp(self):
        # On redéfinit la fonction report pour capturer les messages d'alerte
        self.messages = []
        generated_rules_R8.report = lambda msg: self.messages.append(msg)

    def run_rule(self, code):
        # Parser le code source et exécuter la règle R8 sur l'AST
        ast_node = ast.parse(code)
        generated_rules_R8.rule_R8(ast_node)

    def test_detect_pytorch_call_method_misuse(self):
        """Test du cas où l'appel direct à forward() est utilisé (mauvaise pratique)"""
        code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Mauvaise pratique : utilisation directe de forward()
        x = self.pool.forward(F.relu(self.conv1(x)))
        return x
"""
        self.run_rule(code)
        self.assertGreater(len(self.messages), 0, "Une alerte devrait être générée pour l'utilisation de .forward() directement")

    def test_detect_pytorch_call_method_correct(self):
        """Test du cas où l'appel est fait correctement avec self.pool()"""
        code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Bonne pratique : appel direct au module (self.pool())
        x = self.pool(F.relu(self.conv1(x)))
        return x
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Aucune alerte ne devrait être générée lorsque le module est appelé correctement")

if __name__ == '__main__':
    unittest.main()
