import ast
import unittest
import generated_rules_3  # module généré contenant la règle R3

class TestTensorArrayNotUsedR3(unittest.TestCase):
    def setUp(self):
        # Réinitialise les messages pour chaque test
        self.messages = []
        # Remplace la fonction report pour capturer les messages d'alerte
        generated_rules_3.report = lambda msg: self.messages.append(msg)

    def run_rule(self, code):
        # Parse le code source et exécute la règle R3
        ast_node = ast.parse(code)
        generated_rules_3.rule_R3(ast_node)

    def test_detect_constant_concat_in_tf_function(self):
        # Cas de test avec tf.constant() et tf.concat() dans une fonction tf.function problématique
        code = """
import tensorflow as tf

@tf.function
def problematic_function():
    a = tf.constant([1, 2])
    for i in range(3):
        a = tf.concat([a, [i]], 0)
    return a
"""
        self.run_rule(code)
        self.assertGreater(len(self.messages), 0, "Should detect missing TensorArray usage in problematic_function")

    def test_correct_tensorarray_usage(self):
        # Cas de test avec une utilisation correcte de TensorArray
        code = """
import tensorflow as tf

@tf.function
def correct_function():
    a = tf.TensorArray(tf.int32, size=5)
    a = a.write(0, 1)
    a = a.write(1, 2)
    return a.stack()
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should not detect issue in correct_function")

    def test_detect_multiple_constant_concat(self):
        # Cas de test avec plusieurs variables utilisant constant et concat
        code = """
import tensorflow as tf

@tf.function
def multiple_variables_function():
    x = tf.constant([1, 2])
    y = tf.constant([3, 4])
    for i in range(3):
        x = tf.concat([x, [i]], 0)
        y = tf.concat([y, [i+10]], 0)
    return x, y
"""
        self.run_rule(code)
        self.assertGreater(len(self.messages), 0, "Should detect issues in multiple_variables_function")

    def test_nested_function_detection(self):
        # Cas de test dans une fonction imbriquée avec @tf.function
        code = """
import tensorflow as tf

@tf.function
def outer_function():
    def inner_function():
        a = tf.constant([1, 2])
        for i in range(3):
            a = tf.concat([a, [i]], 0)
        return a
    return inner_function()
"""
        self.run_rule(code)
        self.assertGreater(len(self.messages), 0, "Should detect issue in inner_function nested within outer_function")

if __name__ == '__main__':
    unittest.main()
