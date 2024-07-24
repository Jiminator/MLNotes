# Roll Again 1

## Question 1
Kelly rolls a fair standard 6-sided die. She observes the value on the face. Afterwards, she is given the option to either receive the upface (in dollars) as a payout or to roll again. If she rolls again, she receives the upface of the second roll (in dollars) as payout, regardless of what she rolled on the first turn. If Kelly plays optimally, what is her expected payout?

### Answer
The expected value of 1 roll is 

$$ E[\text{1 roll}] = \frac{1}{6} \cdot 1 + \frac{1}{6} \cdot 2 + \frac{1}{6} \cdot 3 + \frac{1}{6} \cdot 4 + \frac{1}{6} \cdot 5 + \frac{1}{6} \cdot 6 = 3.5$$

Kelly's optimal playing strategy is to reroll if she rolls a value lower than the expected value of 1 roll (1,2, or 3), and to cash-in on the first roll otherwise (rolling a 4,5, or 6). This is because if she chooses to roll again, she can expect to receive a payout of 3.5 in the second roll.

The event tree is shown below:
![alt text](image.png)


Thus, The expected value of such as strategy is shown below:

$$E[\text{optimal}] = \frac{1}{6} \cdot E[\text{1 roll}] + \frac{1}{6} \cdot E[\text{1 roll}] + \frac{1}{6} \cdot E[\text{1 roll}] + \frac{1}{6} \cdot 4 + \frac{1}{6} \cdot 5 + \frac{1}{6} \cdot 6 = \boxed{4.25}$$



<details>
  <summary>Code for Graph</summary>

  ### Code
  ```python
import matplotlib.pyplot as plt
import networkx as nx
import fractions

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges for the first roll
G.add_node("Start")
for i in range(1, 7):
    first_roll = f"{i}"
    G.add_node(first_roll)
    G.add_edge("Start", first_roll, weight=1/6)

    # Add nodes and edges for the second roll if the first roll is less than 3.5
    if i <= 3:
        for j in range(1, 7):
            second_roll = f"{i}, {j}"
            G.add_node(second_roll)
            G.add_edge(first_roll, second_roll, weight=1/6)

# Adjust positions to space out nodes and reduce clutter
pos = {
    "Start": (0, 0),
    "1": (-8, -1),
    "2": (-4, -1),
    "3": (0, -1),
    "4": (4, -1),
    "5": (8, -1),
    "6": (12, -1)
}
for i in range(1, 7):
    pos[f"1, {i}"] = (pos["1"][0] + (i-3.5) * 1.5, -2)
    pos[f"2, {i}"] = (pos["2"][0] + (i-3.5), -2)
    pos[f"3, {i}"] = (pos["3"][0] + (i-3.5) * 1.5, -2)

# Redraw the graph with adjusted positions and fractional probabilities
plt.figure(figsize=(14, 7))

# Draw the nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

# Draw the edges with fractional labels
edges = G.edges(data=True)
nx.draw_networkx_edges(G, pos, edgelist=edges)
edge_labels = {(u, v): f"{fractions.Fraction(d['weight']).limit_denominator()}" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Draw the labels for nodes
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# Set title
plt.title("Probability Tree")

# Show plot
plt.show()
  ```
</details>

## Question 2
A frog starts at the center of a circular lily pad of radius 1 meter. The frog hops to a uniformly random point on the lily pad. Find the expected radial distance of the frog from the center of the lily pad.

### Answer
Let $R$ denote a random variable that is the radial distance of the frog from the center of the circle. We would like to find the probability density function of R, which requires us to first find the cumulative distribution of R. To find this, we observe that the probability that the frog lands within a certain region of the circle is proportional to the area of that region.

$$F_R(r) = P(R \leq r) = \frac{\text{Area of circle of radius }r}{\text{Total area}} = \frac{\pi r^2}{\pi \cdot 1^2} = r^2 \text{ for } 0 \leq r \leq 1$$

The PDF $f_R(r)$ is the derivative of the CDF $F_R(r)$:
$$f_R(r) = \frac{d}{dr}F_R(r) = \frac{d}{dr}r^2 = 2r \text{ for } 0 \leq r \leq 1$$

The expected value $E[R]$ is the integral of $r$ times the PDF over the interval from 0 to 1:
$$E[R] = \int_0^1 r\cdot f_R(r)dr = \int^1_0 r\cdot 2rdr = 2 \int^1_0 r^2dr = 2\left[\frac{r^3}{3} \right]^1_0 = 2 \cdot \frac{1^3}{3} = \boxed{\frac{2}{3}}$$

## Question 3
$n$ friends sit around a circular table of $n$ seats uniformly at random. Assume that each of the friends has a distinct age. Find the probability that they sit down at the table in age order when $n = 6$. This means that either clockwise or counter-clockwise, the ages of the people are in strictly increasing order.

### Answer
We first need to find the total number of permutations. For $n$ distinct individuals, the number of ways they can sit in a line is $n!$. However, since the arrangement is circular, we have to account for rotation symmetry. The number of distinct arrangements is $\frac{n!}{n} = (n-1)!$ The key intuition is that in circular permutations, the first person is considered a place holder, and where he sits does not matter. In another words, if we put one person at the "top" of the table, the the others can permute linearly ($n-1$) permutations. Another way to think about it that each circular arrangement of $n$ people corresponds to $n$ linear arrangements since for each circular arrangement we can rotate $n$ times clockwise and still have the same configuration.

Now that we found the number of distinct permutations, the questions becomes quite trivial as tehere are only two acceptable arrangements for age order (clockwise and counter-clockwise). The answer is thus $$\frac{2}{(n-1)!} = \frac{2}{5!} = \frac{2}{120} = \boxed{\frac{1}{60}}$$