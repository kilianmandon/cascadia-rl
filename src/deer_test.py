from scoring import bruteforce_form_fit


def main():
    nodes = [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 2),
        (3, 2)
    ]

    scores = [2, 5, 9, 13]
    lines = [
        [(0, i) for i in range(length)]
        for length in range(1, 5)
    ]

    score = bruteforce_form_fit(nodes, lines, scores)
    print(score)

if __name__=='__main__':
    main()