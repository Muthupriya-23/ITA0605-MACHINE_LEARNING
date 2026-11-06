
import csv

a = []

with open('enjoysport.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader, None)

    for row in reader:
        if row and len(row) == 7:
            a.append(row)

print("Training data:", a)
print("\nThe total number of training instances are:", len(a))

num_attribute = len(a[0]) - 1

print("\nThe initial hypothesis is:")
hypothesis = ['0'] * num_attribute
print(hypothesis)

for i in range(len(a)):
    if a[i][num_attribute].strip().lower() == 'yes':
        for j in range(num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'

    print("\nThe hypothesis for training instance", i + 1, "is:")
    print(hypothesis)

print("\nThe maximally specific hypothesis is:")
print(hypothesis)

