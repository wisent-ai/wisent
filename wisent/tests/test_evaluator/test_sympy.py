from sympy.parsing.latex import parse_latex

expr1 = parse_latex(r"\frac{1}{2} + \sqrt{3}")

value = expr1 + 1

print(str(value))

print(float(value))

expr2 = parse_latex("\left( 3, \frac{\pi}{2} \right))")
print(expr2)


