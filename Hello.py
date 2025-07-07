
# Variable Type and Classes, switching between them
"""height = 180.5
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
"""
# functions - code that's packaged nicely and can be called when needed
def hello(name, age):
    print('Hi there ' + name + '!')
    print('Happy ' + str(age) + 'th birthday!')
my_name = input('What is your name?: ')
my_age = input('What is your age?: ')
hello(my_name, my_age)
