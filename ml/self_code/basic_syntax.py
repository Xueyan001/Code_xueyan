"""python pandas"""
#string_variable_2 = "he said 'OK'"
#print("",string_variable_2)
#long_string = 3 * "3"
#print("", long_string)
# list_variable_3 = [1,2,"d",[1,2]]
# print("",list_variable_3)
dict = {"a":1, "b":2, "c":3} #dictionaries-->key:value
value = dict.get("c", 0) #get with 0 as default, or the value of the key " "
del dict["a"]
key_list = list(dict.keys())
key_values = list(dict.values())
print(key_list)
print(key_values)
# tuple_variable = (1,2,3)
# result = tuple_variable[0] == 3 # == means compare, output the booleans
# print("", result)
# list_variable = [1,2,3]
# list_variable.append(4) # add 4 as the last element
# list_variable.reverse() # reverse the list
# # list_variable[0] = 4
# print("", list_variable)

 
x = 42
print(x == 42) # equals, True
print(x != 42) # not equal, False
print(x > 42) # greater than, False
print(x >= 42) # greater than or equal to, True
print(x < 42) # less than, False
print(x <= 42) # less than or equal to, True
# you can combine several conditions
print((x <= 42 and x > 42) or x == 42) # True
