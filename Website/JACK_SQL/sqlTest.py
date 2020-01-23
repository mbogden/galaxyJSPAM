import mysql.connector
import os
from pathlib import Path

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="password",
  database="GALAXY"
)

# Buffered is apparantly important 
cursor = mydb.cursor(buffered=True)

def showDB():
    query = ("SELECT * FROM WEB;")

    cursor.execute(query)

    for (sdss, path) in cursor:
        print(sdss, path)

# Reads the directories from the fed in path
# Fills connected db with rows of (sdss, path/to/target_zoo.png)
def fillDB(dirIn):
    from glob import glob
    
    for g in glob(dirIn):
        if os.path.isfile(g + "sdssParameters/target_zoo.png" ):
            cmd = "INSERT INTO WEB (sdss, path) VALUES ('" + Path(g).stem + "', '" + os.path.abspath(g + "sdssParameters/target_zoo.png'")  + ");"
            try:
                cursor.execute(cmd)
            except Exception as e:
                # 1062 is duplicate entry error, don't need cmd
                if e.errno != 1062:
                    print(cmd)
                print(e)

if __name__ == "__main__":
    fillDB("sql_spam_data/*/")
    showDB()

    # Specific query examples
    # May need some type coercion, since tuples are usually the rows
    # Using fetchall() returns: Total count: [(3,)]
    # Using fetchone() instead returns (3,) which is a little easier for some uses

    # How many in db?
    cursor.execute("SELECT COUNT(*) FROM WEB")
    print("Total count:", cursor.fetchone()[0])

    # Get all the sdss's we know of
    cursor.execute("SELECT sdss FROM WEB")
    print("Known sdss's:", ", ".join([x[0] for x in cursor]))

    # Get the zoo image from a known sdss
    # Assuming we get the sdss froms somewhere else, like a label on the website
    sdss = "1237678620102623480"
    cursor.execute("SELECT path FROM WEB WHERE sdss = " + sdss)
    print("Path: ", cursor.fetchone()[0])

    mydb.commit()
    cursor.close()
    mydb.close()





