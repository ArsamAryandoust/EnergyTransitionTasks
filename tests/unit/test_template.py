import unittest
import sys
sys.path.append('/EnergyTransitionTasks/src')

class TestSomeFunctions(unittest.TestCase):
    """ 
    Tests functions defined in single .py file
    """

    @classmethod
    def setUpClass(cls):
        """ 
        Runs once before the first test.
        """

        print('Unit test template ran correctly for code before first test!')


    @classmethod
    def tearDownClass(cls):
        """ 
        Runs once after the last test.
        """

        print('Unit test template ran correctly for code after last test!')


    def setUp(self):
        """ 
        Runs before every test.
        """
        
        print('Unit test template ran correctly for code before every test!')
        

    def tearDown(self):
        """ 
        Runs after every test.
        """
        
        print('Unit test template ran correctly for code after every test!')
        
        
    def test_templatefunction_1(self):
        print(
            'Unit test test_templatefunction_1 ran correctly from',
            'test/unit/test_template.py inside Docker!'
        )
        
    def test_templatefunction_2(self):
        print(
            'Unit test test_templatefunction_2 ran correctly from',
            'test/unit/test_template.py inside Docker!'
        )


if __name__ == '__main__':
    """
    Executes every method defined in classes herited from unittest.TestCase and
    corresponding functions before and after.
    """
    unittest.main()

