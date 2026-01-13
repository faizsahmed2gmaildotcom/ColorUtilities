my_str = "Caf\xc3\xa9 au lait"
print(type(my_str))
print(f"{my_str.encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8').encode('latin1').decode('utf-8').replace(' ', '_').lower()} = {67}")
