import json
import random
import os
from typing import List, Dict, Tuple
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
import argparse


class MathProblemGenerator:
    """Generate mathematical problems for RPT training"""
    
    def __init__(self):
        self.problem_types = [
            self.generate_arithmetic,
            self.generate_algebra,
            self.generate_word_problem,
            self.generate_geometry,
            self.generate_number_theory,
            self.generate_calculus_basic,
            self.generate_probability,
            self.generate_combinatorics
        ]
    
    def generate_arithmetic(self) -> Dict[str, str]:
        """Generate basic arithmetic problems"""
        ops = ['+', '-', '*', '/']
        op = random.choice(ops)
        
        if op == '/':
            b = random.randint(1, 20)
            a = b * random.randint(1, 20)
        else:
            a = random.randint(1, 100)
            b = random.randint(1, 100)
        
        question = f"Calculate {a} {op} {b}"
        
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        elif op == '*':
            answer = a * b
        else:
            answer = a / b
            
        solution = f"To solve {a} {op} {b}:\n"
        if op == '+':
            solution += f"{a} + {b} = {answer}"
        elif op == '-':
            solution += f"{a} - {b} = {answer}"
        elif op == '*':
            solution += f"{a} × {b} = {answer}"
        else:
            solution += f"{a} ÷ {b} = {answer}"
            
        return {
            "question": question,
            "answer": str(answer),
            "solution": solution,
            "difficulty": "easy"
        }
    
    def generate_algebra(self) -> Dict[str, str]:
        """Generate algebra problems"""
        problem_type = random.choice(['linear', 'quadratic', 'system'])
        
        if problem_type == 'linear':
            a = random.randint(2, 10)
            b = random.randint(-20, 20)
            c = random.randint(-50, 50)
            
            question = f"Solve for x: {a}x + {b} = {c}"
            x = (c - b) / a
            
            solution = f"To solve {a}x + {b} = {c}:\n"
            solution += f"1. Subtract {b} from both sides: {a}x = {c - b}\n"
            solution += f"2. Divide by {a}: x = {c - b}/{a} = {x}"
            
            answer = str(x)
            
        elif problem_type == 'quadratic':
            a = 1
            b = random.randint(-10, 10)
            c = random.randint(-25, 25)
            
            discriminant = b**2 - 4*a*c
            
            question = f"Solve for x: x² + {b}x + {c} = 0"
            
            if discriminant >= 0:
                x1 = (-b + discriminant**0.5) / (2*a)
                x2 = (-b - discriminant**0.5) / (2*a)
                
                solution = f"Using the quadratic formula:\n"
                solution += f"x = (-b ± √(b² - 4ac)) / 2a\n"
                solution += f"x = ({-b} ± √({b}² - 4(1)({c}))) / 2\n"
                solution += f"x = ({-b} ± √{discriminant}) / 2\n"
                solution += f"x₁ = {x1}, x₂ = {x2}"
                
                answer = f"x₁ = {x1}, x₂ = {x2}"
            else:
                solution = f"The discriminant b² - 4ac = {discriminant} < 0\n"
                solution += "Therefore, there are no real solutions."
                answer = "No real solutions"
                
        else:  # system
            a1, b1 = random.randint(1, 5), random.randint(1, 5)
            a2, b2 = random.randint(1, 5), random.randint(1, 5)
            
            # Ensure non-parallel lines
            while a1*b2 == a2*b1:
                a2, b2 = random.randint(1, 5), random.randint(1, 5)
                
            c1 = random.randint(5, 20)
            c2 = random.randint(5, 20)
            
            question = f"Solve the system:\n{a1}x + {b1}y = {c1}\n{a2}x + {b2}y = {c2}"
            
            # Solve using elimination
            det = a1*b2 - a2*b1
            x = (c1*b2 - c2*b1) / det
            y = (a1*c2 - a2*c1) / det
            
            solution = f"Using elimination method:\n"
            solution += f"Multiply first equation by {b2} and second by {b1}:\n"
            solution += f"{a1*b2}x + {b1*b2}y = {c1*b2}\n"
            solution += f"{a2*b1}x + {b2*b1}y = {c2*b1}\n"
            solution += f"Subtract: {det}x = {c1*b2 - c2*b1}\n"
            solution += f"Therefore: x = {x}, y = {y}"
            
            answer = f"x = {x}, y = {y}"
            
        return {
            "question": question,
            "answer": answer,
            "solution": solution,
            "difficulty": "medium"
        }
    
    def generate_word_problem(self) -> Dict[str, str]:
        """Generate word problems"""
        templates = [
            {
                "template": "A train travels {speed} km/h for {time} hours. How far does it travel?",
                "vars": {"speed": (50, 150), "time": (1, 8)},
                "calc": lambda v: v["speed"] * v["time"],
                "solution": "Distance = Speed × Time = {speed} × {time} = {result} km"
            },
            {
                "template": "A store offers a {discount}% discount on an item priced at ${price}. What is the final price?",
                "vars": {"discount": (10, 40), "price": (20, 200)},
                "calc": lambda v: v["price"] * (1 - v["discount"]/100),
                "solution": "Final price = Original price × (1 - discount) = {price} × (1 - {discount}/100) = ${result:.2f}"
            },
            {
                "template": "If {workers} workers can complete a job in {days} days, how many days would it take {new_workers} workers?",
                "vars": {"workers": (4, 12), "days": (5, 20), "new_workers": (6, 15)},
                "calc": lambda v: (v["workers"] * v["days"]) / v["new_workers"],
                "solution": "Work = {workers} × {days} = {work} worker-days\nTime for {new_workers} workers = {work} ÷ {new_workers} = {result:.2f} days"
            }
        ]
        
        template = random.choice(templates)
        vars = {k: random.randint(*v) for k, v in template["vars"].items()}
        
        question = template["template"].format(**vars)
        result = template["calc"](vars)
        
        # Calculate intermediate values for solution
        if "work" in template["solution"]:
            vars["work"] = vars["workers"] * vars["days"]
        vars["result"] = result
        
        solution = template["solution"].format(**vars)
        
        return {
            "question": question,
            "answer": str(result),
            "solution": solution,
            "difficulty": "medium"
        }
    
    def generate_geometry(self) -> Dict[str, str]:
        """Generate geometry problems"""
        shapes = ['circle', 'rectangle', 'triangle', 'square']
        shape = random.choice(shapes)
        
        if shape == 'circle':
            radius = random.randint(2, 20)
            question = f"Find the area of a circle with radius {radius} units."
            area = 3.14159 * radius ** 2
            
            solution = f"Area of circle = πr²\n"
            solution += f"Area = π × {radius}² = π × {radius**2} ≈ {area:.2f} square units"
            
            answer = f"{area:.2f}"
            
        elif shape == 'rectangle':
            length = random.randint(5, 30)
            width = random.randint(5, 30)
            question = f"Find the perimeter of a rectangle with length {length} and width {width}."
            perimeter = 2 * (length + width)
            
            solution = f"Perimeter of rectangle = 2(length + width)\n"
            solution += f"Perimeter = 2({length} + {width}) = 2 × {length + width} = {perimeter} units"
            
            answer = str(perimeter)
            
        elif shape == 'triangle':
            a = random.randint(3, 15)
            b = random.randint(4, 16)
            c = ((a**2 + b**2) ** 0.5)
            
            question = f"A right triangle has legs of length {a} and {b}. Find the hypotenuse."
            
            solution = f"Using Pythagorean theorem: c² = a² + b²\n"
            solution += f"c² = {a}² + {b}² = {a**2} + {b**2} = {a**2 + b**2}\n"
            solution += f"c = √{a**2 + b**2} ≈ {c:.2f}"
            
            answer = f"{c:.2f}"
            
        else:  # square
            side = random.randint(5, 25)
            calc_type = random.choice(['area', 'perimeter'])
            
            if calc_type == 'area':
                question = f"Find the area of a square with side length {side}."
                result = side ** 2
                solution = f"Area of square = side²\nArea = {side}² = {result} square units"
            else:
                question = f"Find the perimeter of a square with side length {side}."
                result = 4 * side
                solution = f"Perimeter of square = 4 × side\nPerimeter = 4 × {side} = {result} units"
                
            answer = str(result)
            
        return {
            "question": question,
            "answer": answer,
            "solution": solution,
            "difficulty": "medium"
        }
    
    def generate_number_theory(self) -> Dict[str, str]:
        """Generate number theory problems"""
        problem_type = random.choice(['prime', 'gcd', 'lcm', 'factorial'])
        
        if problem_type == 'prime':
            n = random.randint(20, 100)
            question = f"Is {n} a prime number?"
            
            is_prime = True
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    is_prime = False
                    break
                    
            if is_prime:
                solution = f"To check if {n} is prime, we test divisibility up to √{n} ≈ {int(n**0.5)}\n"
                solution += f"{n} is not divisible by any number from 2 to {int(n**0.5)}\n"
                solution += f"Therefore, {n} is prime."
                answer = "Yes"
            else:
                solution = f"To check if {n} is prime, we test divisibility\n"
                solution += f"{n} = {i} × {n//i}\n"
                solution += f"Therefore, {n} is not prime."
                answer = "No"
                
        elif problem_type == 'gcd':
            a = random.randint(12, 60)
            b = random.randint(12, 60)
            question = f"Find the GCD of {a} and {b}."
            
            def gcd(x, y):
                while y:
                    x, y = y, x % y
                return x
                
            result = gcd(a, b)
            
            solution = f"Using Euclidean algorithm:\n"
            x, y = a, b
            steps = []
            while y:
                steps.append(f"{x} = {x//y} × {y} + {x%y}")
                x, y = y, x % y
            solution += "\n".join(steps) + f"\nGCD({a}, {b}) = {result}"
            
            answer = str(result)
            
        elif problem_type == 'lcm':
            a = random.randint(4, 20)
            b = random.randint(4, 20)
            question = f"Find the LCM of {a} and {b}."
            
            def gcd(x, y):
                while y:
                    x, y = y, x % y
                return x
                
            lcm = (a * b) // gcd(a, b)
            
            solution = f"LCM(a, b) = (a × b) / GCD(a, b)\n"
            solution += f"GCD({a}, {b}) = {gcd(a, b)}\n"
            solution += f"LCM({a}, {b}) = ({a} × {b}) / {gcd(a, b)} = {a*b} / {gcd(a, b)} = {lcm}"
            
            answer = str(lcm)
            
        else:  # factorial
            n = random.randint(3, 8)
            question = f"Calculate {n}!"
            
            result = 1
            solution = f"{n}! = "
            terms = []
            for i in range(n, 0, -1):
                result *= i
                terms.append(str(i))
            solution += " × ".join(terms) + f" = {result}"
            
            answer = str(result)
            
        return {
            "question": question,
            "answer": answer,
            "solution": solution,
            "difficulty": "medium"
        }
    
    def generate_calculus_basic(self) -> Dict[str, str]:
        """Generate basic calculus problems"""
        problem_type = random.choice(['derivative', 'integral'])
        
        if problem_type == 'derivative':
            power = random.randint(2, 5)
            coeff = random.randint(1, 5)
            
            question = f"Find the derivative of f(x) = {coeff}x^{power}"
            
            solution = f"Using power rule: d/dx(ax^n) = n·a·x^(n-1)\n"
            solution += f"f'(x) = {power} · {coeff} · x^({power}-1)\n"
            solution += f"f'(x) = {power * coeff}x^{power-1}"
            
            answer = f"{power * coeff}x^{power-1}"
            
        else:  # integral
            power = random.randint(1, 4)
            coeff = random.randint(1, 5)
            
            question = f"Find the integral of f(x) = {coeff}x^{power}"
            
            solution = f"Using power rule: ∫ax^n dx = a·x^(n+1)/(n+1) + C\n"
            solution += f"∫{coeff}x^{power} dx = {coeff}·x^({power}+1)/({power}+1) + C\n"
            solution += f"∫{coeff}x^{power} dx = {coeff}x^{power+1}/{power+1} + C"
            
            if coeff % (power + 1) == 0:
                simplified_coeff = coeff // (power + 1)
                solution += f" = {simplified_coeff}x^{power+1} + C"
                answer = f"{simplified_coeff}x^{power+1} + C"
            else:
                answer = f"{coeff}x^{power+1}/{power+1} + C"
                
        return {
            "question": question,
            "answer": answer,
            "solution": solution,
            "difficulty": "hard"
        }
    
    def generate_probability(self) -> Dict[str, str]:
        """Generate probability problems"""
        problem_type = random.choice(['dice', 'cards', 'coins'])
        
        if problem_type == 'dice':
            n_dice = random.randint(2, 3)
            target = random.randint(n_dice * 2, n_dice * 5)
            
            question = f"What is the probability of rolling a sum of {target} with {n_dice} dice?"
            
            # Count favorable outcomes
            favorable = 0
            total = 6 ** n_dice
            
            # Simple case for 2 dice
            if n_dice == 2:
                for i in range(1, 7):
                    for j in range(1, 7):
                        if i + j == target:
                            favorable += 1
                            
                solution = f"Total outcomes with {n_dice} dice: 6^{n_dice} = {total}\n"
                solution += f"Favorable outcomes (sum = {target}): {favorable}\n"
                solution += f"Probability = {favorable}/{total}"
                
                if favorable > 0:
                    from fractions import Fraction
                    frac = Fraction(favorable, total)
                    solution += f" = {frac}"
                    answer = str(frac)
                else:
                    answer = "0"
            else:
                # Simplified for 3 dice
                solution = f"With {n_dice} dice, we need to count all ways to get sum {target}\n"
                solution += f"This is a complex calculation involving partitions.\n"
                solution += f"Total outcomes = 6^{n_dice} = {total}"
                answer = "Requires detailed enumeration"
                
        elif problem_type == 'cards':
            suits = ['hearts', 'diamonds', 'clubs', 'spades']
            suit = random.choice(suits)
            
            question = f"What is the probability of drawing a {suit} from a standard deck?"
            
            solution = f"Standard deck has 52 cards\n"
            solution += f"Each suit has 13 cards\n"
            solution += f"Probability = 13/52 = 1/4"
            
            answer = "1/4"
            
        else:  # coins
            n_flips = random.randint(3, 5)
            n_heads = random.randint(1, n_flips - 1)
            
            question = f"What is the probability of getting exactly {n_heads} heads in {n_flips} coin flips?"
            
            from math import comb
            favorable = comb(n_flips, n_heads)
            total = 2 ** n_flips
            
            solution = f"Using binomial probability:\n"
            solution += f"Number of ways to get {n_heads} heads in {n_flips} flips = C({n_flips},{n_heads}) = {favorable}\n"
            solution += f"Total outcomes = 2^{n_flips} = {total}\n"
            solution += f"Probability = {favorable}/{total}"
            
            from fractions import Fraction
            frac = Fraction(favorable, total)
            solution += f" = {frac}"
            
            answer = str(frac)
            
        return {
            "question": question,
            "answer": answer,
            "solution": solution,
            "difficulty": "hard"
        }
    
    def generate_combinatorics(self) -> Dict[str, str]:
        """Generate combinatorics problems"""
        problem_type = random.choice(['permutation', 'combination'])
        
        n = random.randint(5, 10)
        r = random.randint(2, min(5, n-1))
        
        if problem_type == 'permutation':
            question = f"How many ways can you arrange {r} items from a set of {n} items?"
            
            from math import factorial
            result = factorial(n) // factorial(n - r)
            
            solution = f"Permutations P(n,r) = n!/(n-r)!\n"
            solution += f"P({n},{r}) = {n}!/({n}-{r})!\n"
            solution += f"P({n},{r}) = {n}!/{n-r}!\n"
            solution += f"P({n},{r}) = {result}"
            
            answer = str(result)
            
        else:  # combination
            question = f"How many ways can you choose {r} items from a set of {n} items?"
            
            from math import comb
            result = comb(n, r)
            
            solution = f"Combinations C(n,r) = n!/(r!(n-r)!)\n"
            solution += f"C({n},{r}) = {n}!/({r}!×{n-r}!)\n"
            solution += f"C({n},{r}) = {result}"
            
            answer = str(result)
            
        return {
            "question": question,
            "answer": answer,
            "solution": solution,
            "difficulty": "hard"
        }
    
    def generate_problem(self) -> Dict[str, str]:
        """Generate a random math problem"""
        generator = random.choice(self.problem_types)
        return generator()


def create_math_dataset(
    output_path: str,
    num_problems: int = 1000,
    difficulty_distribution: Dict[str, float] = None
):
    """Create a mathematical dataset for RPT training"""
    
    if difficulty_distribution is None:
        difficulty_distribution = {
            "easy": 0.3,
            "medium": 0.5,
            "hard": 0.2
        }
    
    generator = MathProblemGenerator()
    problems = []
    
    logger.info(f"Generating {num_problems} math problems...")
    
    # Generate problems
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    
    for i in range(num_problems):
        problem = generator.generate_problem()
        problems.append(problem)
        difficulty_counts[problem["difficulty"]] += 1
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1} problems...")
    
    # Shuffle problems
    random.shuffle(problems)
    
    # Split into train/val/test
    train_split = int(0.8 * len(problems))
    val_split = int(0.9 * len(problems))
    
    train_problems = problems[:train_split]
    val_problems = problems[train_split:val_split]
    test_problems = problems[val_split:]
    
    # Save datasets
    os.makedirs(output_path, exist_ok=True)
    
    # Save in JSONL format
    for split_name, split_data in [
        ("train", train_problems),
        ("val", val_problems),
        ("test", test_problems)
    ]:
        output_file = os.path.join(output_path, f"{split_name}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for problem in split_data:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(split_data)} problems to {output_file}")
    
    # Save statistics
    stats = {
        "total_problems": num_problems,
        "difficulty_distribution": difficulty_counts,
        "splits": {
            "train": len(train_problems),
            "val": len(val_problems),
            "test": len(test_problems)
        }
    }
    
    stats_file = os.path.join(output_path, "dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset statistics saved to {stats_file}")
    logger.info("Dataset creation complete!")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Create mathematical dataset for RPT training")
    parser.add_argument('--output_path', type=str, default='data/math',
                        help='Output directory for dataset')
    parser.add_argument('--num_problems', type=int, default=4428,
                        help='Number of problems to generate (default: 4428 like OmniMATH)')
    parser.add_argument('--easy_ratio', type=float, default=0.3,
                        help='Ratio of easy problems')
    parser.add_argument('--medium_ratio', type=float, default=0.5,
                        help='Ratio of medium problems')
    parser.add_argument('--hard_ratio', type=float, default=0.2,
                        help='Ratio of hard problems')
    
    args = parser.parse_args()
    
    # Normalize difficulty distribution
    total_ratio = args.easy_ratio + args.medium_ratio + args.hard_ratio
    difficulty_distribution = {
        "easy": args.easy_ratio / total_ratio,
        "medium": args.medium_ratio / total_ratio,
        "hard": args.hard_ratio / total_ratio
    }
    
    # Create dataset
    stats = create_math_dataset(
        output_path=args.output_path,
        num_problems=args.num_problems,
        difficulty_distribution=difficulty_distribution
    )
    
    print("\nDataset created successfully!")
    print(f"Location: {args.output_path}")
    print(f"Total problems: {stats['total_problems']}")
    print(f"Distribution: {stats['difficulty_distribution']}")
    print(f"Splits: {stats['splits']}")


if __name__ == "__main__":
    main()