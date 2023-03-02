import sqlite3

# Connect to the database
conn = sqlite3.connect('db.sqlite3')

# Create a cursor object
cursor = conn.cursor()

# Query the sqlite_master table for table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

# Fetch the results
table_names = cursor.fetchall()

# Print the table names
for name in table_names:
    print(name[0])


cursor.execute('SELECT * FROM auth_user')

# Fetch the results
results = cursor.fetchall()

# Print the results
for row in results:
    print(row)

# Close the connection
conn.close()