import random

numbers = '123456789'
ops = '+ - * % //'.split(' ')


def make_basic(hard):
    expr = random.choice(numbers)
    for i in range(random.randint(0, hard)):
        expr = f'{expr} {random.choice(ops)} {random.choice(numbers)}'
    return expr


def make_full(hard):
    while True:
        try:
            expr = make_basic(hard)
            for i in range(random.randint(0, hard)):
                expr = f'({expr}) {random.choice(ops)} ({make_basic(hard)})'
            expr = f'{expr} = {eval(expr)}X'
            return expr
        except:
            pass


def gentokens(ntok):
    alle = 'X'
    while len(alle) < ntok:
        e = make_full(2)
        alle = alle + e
    return [ord(_) for _ in alle]


if __name__ == '__main__':
    print(gentokens(35))
