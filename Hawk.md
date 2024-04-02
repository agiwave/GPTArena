# Formula

```
Parallel-RNN formula:
s(1) = a(1) * s(0) + b(1)
s(2) = a(1) * a(2) * s(0) + a(2)*b(1) + b(2)
s(n) = a(n) * s(n-1) + b(n)
     = a(1) *...* a(n) * s(0) + a(2) *...*a(n) * b(1) + .... + a(n-1) * b(n-1) + b(n)
cuma = [a(1), a(1) * a(2), ..., a(1)*...*a(n)] = np.cumprod(a)
shifta = [ 1., cuma(1), cuma(2), ...., cuma(n-1)] = 
shiftb = [ s(0), b(1), ..., b(n-1)]
s(n) = cuma(n) * ( shiftb(1) / shifta(1) + shiftb(2) / shifta(2) + .... + shiftb(n) / shifta(n)) + b(n)
```
