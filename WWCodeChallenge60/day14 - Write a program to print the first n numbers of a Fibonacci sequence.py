# Write a program to print the first n numbers of a Fibonacci sequence

def fibonacci(n):
  seq = [0, 1]
  for i in range(2, n):
    seq.append(seq[-1] + seq[-2])
  return seq

try:
  num = int(input(f'Enter a number: '))
  print(f'First {num} numbers of Fibonacci sequence: {fibonacci(num)}')

except ValueError:
  print(f'Invalid input')
