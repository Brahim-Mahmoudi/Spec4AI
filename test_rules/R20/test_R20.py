import ast
import unittest
import generated_rules_R20

class TestGeneratedRules20(unittest.TestCase):
    def setUp(self):
        # On réinitialise la liste des messages pour chaque test
        self.messages = []
        def report(message):
            self.messages.append(message)
        # On "monkey-patche" la fonction report du module généré
        generated_rules_R20.report = report

    def run_rule(self, code):
        ast_node = ast.parse(code)
        generated_rules_R20.rule_R20(ast_node)

    def test_loc_usage(self):
        # Test 1 : Utilisation de loc (indexation correcte)
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
result = df.loc[:, 'A']
"""
        self.run_rule(code)
        # Aucune alerte attendue
        self.assertEqual(len(self.messages), 0,
                         "Aucune alerte ne doit être générée pour l'utilisation de loc.")

    def test_chained_indexing(self):
        # Test 2 : Chained indexing (indexation enchaînée)
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
result = df['A']['B']
"""
        self.run_rule(code)
        # Au moins un message doit être généré
        self.assertTrue(len(self.messages) > 0,
                        "Une alerte doit être générée pour le chained indexing.")
        # Afficher le message généré pour le débogage
        print("Message généré:", self.messages[0])

    def test_non_dataframe(self):
        # Test 3 : Utilisation d'un tableau 2D (non DataFrame)
        code = """
arr = [[1, 2, 3], [4, 5, 6]]
result = arr[0][1]
"""
        self.run_rule(code)
        # Aucune alerte attendue pour un tableau qui n'est pas un DataFrame
        self.assertEqual(len(self.messages), 0,
                         "Aucune alerte ne doit être générée pour un tableau non DataFrame.")

if __name__ == '__main__':
    unittest.main()
