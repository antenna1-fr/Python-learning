data = []
import random
import csv

for i in range(500000):
    datapoint = {'day_number': i, 'temperature': random.random()*35, 'precipitation': random.random()*100}
    data.append(datapoint)
with open('large_data_B.csv', mode='w', newline='') as csvfile:
    fieldnames = ['day_number', 'temperature', 'precipitation']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for datapoint in data:
        writer.writerow(datapoint)