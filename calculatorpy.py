class Calculator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def multiplication(self):
        result = self.x*self.y
        return result
   
    
    def addition(self, x, y):
        result = x + y
        return result
        
print(Calculator.addition(2,3))

