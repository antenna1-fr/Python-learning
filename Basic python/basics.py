
# Variable Type and Classes, switching between them

"""height = 180.5
# Start of session 7/6/2025
Pre = "Your Height Is: "
print(height)
print(type(height))
print(Pre+str(height)+"cm")
human = True
print("are you a human: " + str(human))
print(type(human))

# Multiple Assignment - more than one variable per line
name, age, Cool = "Sig" , 21, True
print (name)
print (age)
print (Cool)


Spongebob = Patrick = Sandy = Squidward = 30


# Strings - manipulating strings of characters
name = "Sig"
print(len(name))
print(name.find("S"))
print(name.capitalize())
print(name.upper())
print(name.lower())
print(name.isdigit())
print(name.isalpha())
print(name.count("g"))
print(name.replace("i","o"))
print(name*3)


# Typecasting - convert data type to another
x = 1
y = 2.0
z = "3"

x = str(x)
y = str(y)
z = int(z)

print(str(x))
print(float(z)*3)

print ("x is " + x)
print ("y is " + y)


# User input
name = input("What is your name?: ")
age = float(input("How old are you: "))
age = age + 1
print("Hello " + name)
print("You will be: "+str(age)+" years old in a year")


# math basic 
import math
pi = 3.14
x = 1
y = 2
z = 3

print(round(pi))
print(math.ceil(pi))
print(math.floor(pi))
print(abs(-pi))
print(pow(pi,2))
print(math.sqrt(pi))
print(max(x,y,z))
print(min(x,y,z))

# String Slicing
full_name = "The Sig"
first_letter = full_name[0]
first_name = full_name[0:3] # can also be [:3]
last_name = full_name[4:]
every_other = full_name[0:7:2]
reversed_name = full_name[::-1]

print(first_letter)
print(first_name)
print(last_name)
print(every_other)
print(reversed_name)

website1 = "https://google.com"
website2 = "https://discord.com"
slice = slice(8,-4)
print(website1[slice])
print(website2[slice])

userwebsite = input("Your website is: ")
print(userwebsite)
print (userwebsite[slice])

# If statements

age = int(input("How old are you?: "))
if age == 100:
    print("You are one century old!")
elif age >= 18:
    print("Your age of " + str(age) + " makes you an adult!")
elif age < 0:
    print("You haven't been born yet!")
else:
    print("Your age of " + str(age) + " makes you a child!")

# Logical Operators

temp = int(input('What is the temperature outside?: '))

if not(temp >= 0  and temp <= 30):
    print('The temperature is bad today! Stay inside!')
elif not(temp <= 0  or temp >= 30):
    print('The temperature is normal today! Go outside!')


# While loops - Executes code until the condition is no longer fulfilled

name = ""
while len(name) == 0: # alternatively, "while not name"
    name = input("enter your name: ")

print('Hello ' + name)

# For loops - Executes a block of code a limited amount of times
import time
for i in range(10):
    print(i+1)

for i in range(11,21,2):
    print(i)

for i in "The Sig":
    print(i)

for seconds in range(10,0,-1):
    print(seconds) 
    time.sleep(1)
print("Boom")

# Nested Loops - inner loops will finish all its iterations before the outer loop finishes one

rows = int(input('how many rows?: '))
columns = int(input('how many columns?: '))
symbol = input('Enter a symbol to use: ')

for i in range(rows):
    for j in range (columns):
        print(symbol, end="")
    print()

# Loop Control Statements - change a loops execution from its normal sequence
while True:
    name = input("Enter your name: ")
    if name != "":
        break

phone_number = "123-456-7890"
for i in phone_number:
    if i == "-":
        continue
    print(i, end="")

for i in range(0,21):
    if i == 13:
        pass
    else:
        print(i)

# lists - used to store multiple items in a single variable
food = ['pizza', 'hamburger','hotdog','spaghetti','pudding']
print(food[0])
food[0] = "sushi"
print(food[0])

for x in food:
    print(x, end=" ")

food.append("Torillas")
food.remove("hotdog")
food.pop(1)
food.insert(0, 'Diet_Coke')
food.sort()
food.clear()

# 2D lists - lists of lists
drinks = ['coffee', 'tea', 'soda']
dinner = ['tacos', 'burritos', 'sushi']
desserts = ['matcha','sorbet','pocky']

food = [drinks, dinner, desserts]

print(food[2][1])

# Tuples - collection which is ordered and unchangeable, used to group together related data

student = ('Sig',69,'Cool')
print(student.count('sig'))
print(student.index('Cool'))

for x in student:
    print(x, end=" ")

# Sets - collection which is unordered, unindexed. No duplicate values
utensils = {'fork', 'knife', 'spoon'}
dishes = {'plate', 'bowl', 'cup', 'knife'}
dinner_table = utensils.union(dishes)
utensils.update(dishes)
utensils.add('napkin')
utensils.remove('spoon')
print(utensils.difference(dishes))
print(utensils.intersection(dishes))
for x in utensils: 
    print(x)

# Dictionaries - a changeable, unordered collection of unique key-value pairs
state_capitals = {'California':'Sacramento',
                  'Texas':'Austin', 
                  'Florida':'Tallahassee'}
# print(state_capitals['Alaska'])  # This will raise a KeyError if 'Alaska' is not in the dictionary
print(state_capitals.get('Alaska'))
print(state_capitals.values())
print(state_capitals.keys())
print(state_capitals.items())

state_capitals.update({'Alaska':'Juneau'})
state_capitals.update({'California' : 'Los Angeles'})
state_capitals.pop('Texas')


for key, value in state_capitals.items():
    print(value + " Is " + key + "'s capital")


# index operator [] -  gives access to a sequence's element (strings, lists, tuples)

name = 'the Sig!'
if (name[0]).islower():
    name = name.capitalize()
first_name = name[:3].upper()
last_name  = name[4:].lower()
last_character = name[-1]
print(first_name)
print(last_name)

# functions - code that's packaged nicely and can be called when needed
def hello(name, age):
    print('Hi there ' + name + '!')
    print('Happy ' + str(age) + 'th birthday!')
my_name = input('What is your name?: ')
my_age = input('What is your age?: ')
hello(my_name, my_age)

# End of session 7/6/2025
# Start of session 7/7/2025

# Return statements - functions send python values or objects back to the caller

def multiply(number1, number2):
    return number1* number2
print(multiply(20,31)) 

# keyword arguments - areguments preceded by an identifier when we poass them to a function. not positional

def hello(first, middle, last):
    print('Hello ' + first + ' ' + middle + ' ' + last)
hello('code','dude','bro')
hello(last='code',first='bro',middle='dude')

# nested function calls - functions calls inside other function calls. Innermost are resolved first.

num = input('enter a whole positive number: ')
num = float(num)
num = abs(num)
num = round(num)
print(num)

print(round(abs(float(input('enter a whole positive number: '))))) # Takes up less lines, but is less readable

# variable scope - the region of a program where a variable is accessible. It is only available in the region its created in
name = "Skib" # global scope, accessible anywhere in the file
def display_name():
    name = "Sig" # Defined in the scope of this function
    print(name) # This will use the local variable if available. (Local, Encolsing, Global, Built-in)
display_name()


# *args = parameter that will pack all arguments into a tuple

def multiply(number1, number2):
    return number1* number2
# print(multiply(20,31, 15)) # Will raise an error because it expects only 2 arguments
def multiplymodular(*args):
    total = 1 
    for i in args:
        total *= i
    return total
print(multiplymodular(20,31, 15))


# **kwargs = parameter that will pack all arguments into a dictionary

def hello(**kwargs):
    print('Hello',end=' ')
    for key, value in kwargs.items():
        print(value, end = ' ')

hello(asfg='Sir', ansd='Sig', asdhsdgnjvc='skib')

# str.format() - optional method that gives users more control when displaying output

animal = 'cow'
item = 'moon'
print('The ' +animal +  ' jumped over the ' + item)
print('The {} jumped over the {}'.format(animal, item)) # Inserts the variables into the string positionally.
print('The {0} jumped over the {1}'.format(item, animal)) # Inserts the variables into the string by index. Backwards in this case
print('The {animal} jumped over the {item}'.format(animal='cow', item='moon')) # inserts the variables into the string by keyword

text = "the {} jumped over the {}"
print(text.format(animal, item)) # Inserts the variables into the string by positional arguments

# str.format p2
name = "Sigwerwerewsdf"

print("Hello, my name is {}. Nice to meet you.".format(name))
print("Hello, my name is {:10}. Nice to meet you.".format(name)) # 10 spaces wide, right aligned, to left align use <, center use ^

#str.format p3

number = 4.1453212
number2 =  1200
print('the number is {:.2f}'.format(number))
print('the number is {:,}'.format(number2))
print('the number is {:b}'.format(number2))
print('the number is {:o}'.format(number2))
print('the number is {:X}'.format(number2))
print('the number is {:E}'.format(number2))

# random module - used to generate "random" numbers
import random 
x = print(random.randint(1,6))
y = random.random()

List = ['rock', 'paper', 'scissors']
z = random.choice(List)

Cards = [1,2,3,4,5,6,7,8,9,'J','Q','K']
random.shuffle(Cards)
print(Cards)

 #Exceptions
try:
    numerator = int(input("Enter a numerator: "))
    denominator = int(input("Enter a denominator: "))
    result = numerator / denominator
except ZeroDivisionError as e:
    print("You can't divide by zero! Klown!")
    print(e)
except ValueError as e:
    print('only numbers skib')
    print(e)
except Exception as e:
    print('something went wrong')
    print(e)
else:
    print(result)
finally:
    print('this will always excecute')

# Simple calculator (Ifs, arithmetic, try/except, string formatting)
operator = input("Enter an operator (+, -, *, /): ")
if operator not in ['+', '-', '*', '/']:
    print("Invalid operator. Please enter one of +, -, *, /")
    exit()
try:
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
except ValueError:
    print('Enter a number')
    exit()

if operator == '+':
    answer =  num1 + num2
elif operator == '-':
    answer = num1 - num2
elif operator == '*':
    answer = num1 * num2
elif operator == '/':
    if num2 != 0:
        answer = num1 / num2
    else:
        answer = "Error: Division by zero"
        exit()
else:
    pass

try:
    print(f"{num1}{operator}{num2} is {answer}")
except NameError:
    print("A value was not defined, please try again with valid numbers and operator")
    exit()

# Shopping cart (while loops, for loops, lists, dicts, tuples, functions)

shopping_cart = []
total_cost = 0.0

while True:
    item = input("Enter an item to add to your cart(q to quit): ")
    if item.lower() == 'q':
        break
    try:
        price = float(input(f"How much does {item} cost?"))
    except ValueError:
        print("Please enter a valid price.")
        continue
    shopping_cart.append((item, price))
    total_cost += price
    print(f"{item} added to your cart.")
print("Your shopping cart:")
shopping_cart.sort(key=lambda x: x[1])
for item, price in shopping_cart:
    print(f"{item:13}: ${price:.2f}")
print(f"Total cost: ${total_cost:.2f}")

# Kwargs and args practice
def modularsum (*args):
    sum = 0 
    for i in args:
        sum += i
    return sum
while True:
    try:
        numbers = input("Enter numbers to sum (separated by spaces, q to quit): ")
        if numbers.lower() == 'q':
            break
        num_list = []
        for num in numbers.split():
            num_list.append(float(num))
        print(f"The sum is: {modularsum(*num_list)}")
        exit()
    except ValueError:
        print("Please enter valid numbers.")

#OOP ***IMPORTANT***
# Classes - blueprints for creating objects. Objects are instances of classes
# Attributes -  what an object has, variables that belong to the class
# Methods - What a class can do, functions that belong to the class

# Example: Car racing sim with attributes and methods.
# Takes inputs for a track, two cars and their specs, then simulates a race between them and crowns a winner.

class Car:
    def __init__(self, make, model, year, color, acceleration, top_speed, current_speed=0, track_completion=0):
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        self.acceleration = acceleration
        self.top_speed = top_speed
        self.current_speed = current_speed
        self.track_completion = track_completion
        
    def test_drive(self):
        print(f'This {self.year} {self.color} {self.make} {self.model} is test driving!')
    def stop(self):
        print(f'This {self.year} {self.color} {self.make} {self.model} has stopped!')
    
import time
# Porsche = (Car('Porsche', '919 Tribute', 2018, 'white and red', 5, 369))
# Audi = (Car('Audi', 'R18', 2016, 'black and white', 6, 330))
class Racetrack:
    def __init__(self, name, length, car_1, car_2):
        self.name = name
        self.length = length
        self.car_1 = car_1
        self.car_2 = car_2
        self.winner = None
    def race_start(self):
        print(f'Racing {self.car_1.make} {self.car_1.model} and {self.car_2.make} {self.car_2.model} on {self.name} track!')
        # Simulate
        while self.car_1.track_completion < self.length and self.car_2.track_completion < self.length:
            self.car_1.current_speed += self.car_1.acceleration
            self.car_2.current_speed += self.car_2.acceleration
            if self.car_1.current_speed > self.car_1.top_speed:
                self.car_1.current_speed = self.car_1.top_speed
            if self.car_2.current_speed > self.car_2.top_speed:
                self.car_2.current_speed = self.car_2.top_speed
            self.car_1.track_completion += self.car_1.current_speed * 0.1
            self.car_2.track_completion += self.car_2.current_speed * 0.1
            print(f'{self.car_1.make} {self.car_1.model} is at {self.car_1.track_completion} meters with speed {self.car_1.current_speed} km/h')
            print(f'{self.car_2.make} {self.car_2.model} is at {self.car_2.track_completion} meters with speed {self.car_2.current_speed} km/h')
            time.sleep(.1)
        if self.car_1.track_completion >= self.length and self.car_2.track_completion >= self.length:
            print("It's a tie!")
        elif self.car_1.track_completion >= self.length:
            winner = self.car_1
        else:
            winner = self.car_2
        exit()
    def win(self, winner):
        print(f'The {self.winner.color}, {self.winner.year}, {self.winner.make} {self.winner.model} wins!!!')
try:
    track_name = str(input("Enter the racetrack name: "))
    track_length = int(input("Enter the racetrack length in meters (default 2000): "))
    Car1_make = input("Enter the first car make (default Porsche): ")
    Car1_model = input("Enter the first car model (default 919 Tribute): ")
    Car1_year = int(input("Enter the first car year (default 2018): "))
    Car1_color = input("Enter the first car color (default white and red): ")
    Car1_acceleration = int(input("Enter the first car acceleration (default 5): "))
    Car1_top_speed = int(input("Enter the first car top speed (default 369): "))
    Car2_make = input("Enter the second car make (default Porsche): ")
    Car2_model = input("Enter the second car model (default 919 Tribute): ")
    Car2_year = int(input("Enter the second car year (default 2016): "))
    Car2_color = input("Enter the second car color (default white and red): ")
    Car2_acceleration = int(input("Enter the second car acceleration (default 7): "))
    Car2_top_speed = int(input("Enter the second car top speed (default 300): "))
    racer1 = Car(Car1_make, Car1_model, Car1_year, Car1_color, 5, 369)
    racer2 = Car(Car2_make, Car2_model, Car2_year, Car2_color, 6, 330)
except ValueError as e:
    print("Invalid input, using default values.")
    racer1 = Car('Porsche', '919 Tribute', 2018, 'white and red', 5, 369)
    racer2 = Car('Audi', 'R18', 2016, 'black and white', 6, 330)
    track_name = "Laguna Seca"
    track_length = 2000

track = Racetrack(track_name, track_length, racer1, racer2)  # 2000 meters
track.race_start()
# End of session 7/7/2025
# Start of session 7/9/2025
# Class variables
class Car:
    wheels = 4 #This is a default value. It applies by default to all instances of the object.
    def __init__(self, color):
        self.color = color # this is a variable for the instance of an item. Its value applies to that instance only 
car1 = Car("White")
car2 = Car("Black")
car2.wheels = 2

print(car1.wheels)
print(car2.wheels)
Car.wheels = 2 # Sets this for the whole class, all instances will use this as default from here on

# Inheritance and multilevel inheritance
class Organism: # Top level class
    alive = True
class Animal(Organism): # Derived from Organism, inherits alive
    def eat(self):
        print ('this animal is eating')
    def sleep (self):
        print('this animal is sleeping')
class Rabbit(Animal): # Derived from Animal and Organism, inherits alive, eat, and sleep.
    def run(self):
        print('this rabbit is running')
    pass
class Fish(Animal):
    def swim(self):
        print('this fish is swimming')
    pass
class Hawk(Animal):
    def fly(self):
        print('this hawk is flying')
    pass

rabbit = Rabbit()
fish = Fish()
hawk = Hawk()
print(rabbit.alive)
hawk.sleep()
rabbit.eat()
hawk.fly()
rabbit.run()
fish.swim()

# Multiple Inheritance

class Prey:
    def flee(self):
        print('This animal flees')
class Predator:
    def hunt(self):
        print("This animal hunts")
class Rabbit(Prey):
    pass
class Hawk(Predator):
    pass
class Fish(Prey, Predator): # Inherits from both parent classes as opposed to only one
    pass
rabbit = Rabbit()
hawk = Hawk()
fish = Fish()
rabbit.flee()
hawk.hunt()
fish.flee()
fish.hunt()
"""
import math
math.exp