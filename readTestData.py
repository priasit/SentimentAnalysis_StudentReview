import re
import csv
import array
import os
import msvcrt as m
import sys
def wait():
    m.getch()
	
count = len(sys.argv)
if count == 1:
	print ("Need review files as arguments to process");
	sys.exit(0)
else:
	print ("Numer of files: ", count)

print ('Argument List:', str(sys.argv))

arguments = len(sys.argv) - 1
position = 1  
while (arguments >= position):  
	print ("parameter %i: %s" % (position, sys.argv[position]))
	if not os.path.isfile(sys.argv[position]):
		print ("file: ", sys.argv[position], " is an invalid file or does not exisit")
		sys.exit(0)
	else:
		print ("file: ", sys.argv[position], " exisits")
	position = position + 1

print ("Reviews files are all fine...")

cleandoc = ""
globalstr = ""
reviewlist = []
finallist = []
heading = 0		

def find_between( s, first, last ):
	global cleandoc
	try:
		#print ("<------------------->")
		start = s.index( first ) + len( first )
		end = s.index( last, start )
		inx = len(s[start:end])
		#print ("Start: ", s[start:+5], "End: ", s[end:+5])
		cleandoc = cleandoc[inx:]
		#print ("List Item:\n", s[start:end], "\n", "New Cleandoc start:\n", cleandoc)
		#print ("List Item:\n", s[start:end], "\n")
		#wait()
		return s[start:end]
	except ValueError:
		return ""
		
def create_csv_prep( s, first, last ):
	global newstr
	try:
		start = s.index( first ) + len( first )
		end = s.index( last, start )
		inx = len(s[start:end])
		newstr = newstr[inx:]
		return s[start:end]
	except ValueError:
		return ""
		
def write_csv_file (sentiment, user, datetime, activity):
	global heading
	try:
		#reader = csv.reader(readFile)

		with open('TestHSReviews.csv', 'a', newline='') as f:
			writer = csv.writer(f)

			if listitem > 0:
				#print (sentiment)		
				writer.writerow([sentiment, user, datetime, activity])
			else:
				if heading == 0:
					writer.writerow(["Sentiment", "User", "Datetime", "Activity"])
					heading = 1
				writer.writerow([sentiment, user, datetime, activity])
			
			f.close();

	except ValueError:
		return ""

arguments = len(sys.argv) - 1
item = 1  
while (arguments >= item):

	filename = sys.argv[item]

	# Using the newer with construct to close the file automatically.
	f = open (filename, "r")
	for line in f:
		line = re.sub(r'[^ a-zA-Z0-9 * &]',"",line)
		line = line.lstrip()
		line = line.rstrip()
		cleandoc = cleandoc + line + '\n'

	substring1 = 'All Categories'
	substring2 = 'Report'
	firstreview = cleandoc[(cleandoc.index(substring1)+len(substring1)):cleandoc.index(substring2)]
	reviewlist.append("*"+firstreview)

	#if firstreview:
		#print (firstreview)


	inx=1
	while len(cleandoc) != 0:
		subs= find_between(cleandoc, 'Report', 'Report')

		if subs:
			#print ("index: ", inx, "--------->\n",subs)
			reviewlist.append("*"+subs)
			inx = inx + 1
		else:
			break
		
	finallist = list(set(reviewlist))

	for listitem in range(len(finallist)): 
			
			
		newstr = finallist[listitem]
		count = 0
		#print ("<-----------------Review List---------------------------->");
		while len(newstr) != 0:
				
			if count < 3:
				subs= create_csv_prep(newstr, '*', '*')
			else:
				subs= create_csv_prep(newstr, '*', '\n')
					

			if subs:
				if count == 0:
					#print ("Sentiment:", listitem, "LineCount:", count, subs)
					#print ("Sentiment:\n", subs)
					sentiment = subs
				elif count == 1:
					#print ("User:", listitem, "LineCount:", count, subs)
					#print ("User:\n", subs)
					user = subs
				elif count == 2:
					#print ("Datetime:", listitem, "LineCount:", count, subs)
					#print ("Datetime:\n", subs)
					datetime = subs
				else:
					#print ("Activity:", listitem, "LineCount:", count, subs)
					#print ("Activity:\n", subs)
					activity = subs
					
				if count == 3:
					#print ("Listitem: ", listitem, "count: ", count, "----->")
					#print (sentiment, "\n", user, "\n", datetime, "\n", activity, "\n")
					#print (".")
					write_csv_file (sentiment, user, datetime, activity)
						
				count = count + 1
			else:
				break
				
	item = item + 1