# Roll Again 1

## Question
Kelly rolls a fair standard 6-sided die. She observes the value on the face. Afterwards, she is given the option to either receive the upface (in dollars) as a payout or to roll again. If she rolls again, she receives the upface of the second roll (in dollars) as payout, regardless of what she rolled on the first turn. If Kelly plays optimally, what is her expected payout?

<details>
  <summary>Answer</summary>

### Answer
The expected value of 1 roll is 

$$ E[\text{1 roll}] = \frac{1}{6} \cdot 1 + \frac{1}{6} \  cdot 2 + \frac{1}{6} \cdot 3 + \frac{1}{6} \cdot 4 + \frac{1}{6} \cdot 5 + \frac{1}{6} \cdot 6 = 3.5$$

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
</details>

# Groggy Froggy
## Question
A frog starts at the center of a circular lily pad of radius 1 meter. The frog hops to a uniformly random point on the lily pad. Find the expected radial distance of the frog from the center of the lily pad.

<details>
  <summary>Answer</summary>

### Answer
Let $R$ denote a random variable that is the radial distance of the frog from the center of the circle. We would like to find the probability density function of R, which requires us to first find the cumulative distribution of R. To find this, we observe that the probability that the frog lands within a certain region of the circle is proportional to the area of that region.

$$F_R(r) = P(R \leq r) = \frac{\text{Area of circle of radius }r}{\text{Total area}} = \frac{\pi r^2}{\pi \cdot 1^2} = r^2 \text{ for } 0 \leq r \leq 1$$

The PDF $f_R(r)$ is the derivative of the CDF $F_R(r)$:
$$f_R(r) = \frac{d}{dr}F_R(r) = \frac{d}{dr}r^2 = 2r \text{ for } 0 \leq r \leq 1$$

The expected value $E[R]$ is the integral of $r$ times the PDF over the interval from 0 to 1:
$$E[R] = \int_0^1 r\cdot f_R(r)dr = \int^1_0 r\cdot 2rdr = 2 \int^1_0 r^2dr = 2\left[\frac{r^3}{3} \right]^1_0 = 2 \cdot \frac{1^3}{3} = \boxed{\frac{2}{3}}$$
</details>


# Birthday Circle
## Question
$n$ friends sit around a circular table of $n$ seats uniformly at random. Assume that each of the friends has a distinct age. Find the probability that they sit down at the table in age order when $n = 6$. This means that either clockwise or counter-clockwise, the ages of the people are in strictly increasing order.

<details>
  <summary>Answer</summary>

### Answer
We first need to find the total number of permutations. For $n$ distinct individuals, the number of ways they can sit in a line is $n!$. However, since the arrangement is circular, we have to account for rotation symmetry. The number of distinct arrangements is $\frac{n!}{n} = (n-1)!$ The key intuition is that in circular permutations, the first person is considered a place holder, and where he sits does not matter. In another words, if we put one person at the "top" of the table, the the others can permute linearly ($n-1$) permutations. Another way to think about it that each circular arrangement of $n$ people corresponds to $n$ linear arrangements since for each circular arrangement we can rotate $n$ times clockwise and still have the same configuration.

Now that we found the number of distinct permutations, the questions becomes quite trivial as tehere are only two acceptable arrangements for age order (clockwise and counter-clockwise). The answer is thus $$\frac{2}{(n-1)!} = \frac{2}{5!} = \frac{2}{120} = \boxed{\frac{1}{60}}$$
</details>

# Cordial Handshake 1
### Question
A business meeting begins with 2 representatives from 5 different companies. If every person at the meeting shakes hands with every other person at the meeting, how many handshakes occur total?

<details>
  <summary>Answer</summary>

### Answer
There are two ways to approach this problem. We can randomly select 1 person at random among $n=10$ people and have him shake hands with $n-1$ other people. This results in $n-1$ handshakes. We then select another person out of the remaining $n-1$ people and have him shake hands with everyone that he has not shaken hands with already. Since he has already only shaken hands with one other person, this results in $n-2$ handshakes. This pattern continues until there is only 1 handshake left to give. The sum is $(n-1)+(n-2)+...1 = \frac{(n)(n-1)}{2} = \frac{10 \cdot 9}{2} = \boxed{45}$

This sum could also be framed like so: each of the $n$ representative needs to shake hands with the other $n-1$ representatives. If we aren't worried about repeated handshakes, the total number of handshakes is $n(n-1)$. Since we are worried about repeated handshakes, we divide the result by 2 to avoid double-counting. Thus, the answer is $\frac{(n)(n-1)}{2} = \frac{10 \cdot 9}{2} = \boxed{45}$
</details>

# St. Petersburg 1
### Question
You are offered to play the following game: You flip a fair coin repeatedly until the first head appears. If the first heads appears on the $n$th flip, you received a payout of $$2^n$. What is the fair value of this game. 

<details>
  <summary>Answer</summary>

### Answer
To find the fair value of this game, we need to compute the expected payout. The expected payout E is the sum of the products of the probability of each outcome and its corresponding payout. The probability of getting the first heads on the $n$-th flip is $(\frac{1}{2})^n$. This is because you need $n-1$ tails followed by 1 head, each with probability $\frac{1}{2}$ The expected value can be expressed as an infinite series.
$$E[X] = \sum_{n=1}^{\infty}2^n\left(\frac{1}{2}\right)^n=\sum_{n=1}^{\infty}\left(\frac{2}{2}\right)^n= \sum_{n=1}^{\infty}1 = \boxed{\infty}$$

</details>

# Greater Dice
### Question
You have two fair dice. One of them is $m$-sided, while the other is $n$-sided. The dice have respective values 1-$m$ and 1-$n$ on the sides. Suppose that $n > m$. Find the probability that the $n$-sided die shows a strictly larger value than the $m$-sided die. Solve this with $m=20$ and $n=30$.

<details>
  <summary>Answer</summary>

### Answer

Let $D_{20}$ be the value on the 20-sided die, and $D_{30}$ be the value on the 30-sided die. We want to calculate teh probability that $D_{30} > D_{20}$. The total number of outcomes when rolling both dice is the product of the number of sides on each die: $20 \times 30 = 600$

We need to count the number of outcomes where the value of the 30-sided die is strictly greater than the value of the 20-sided die. We can do this by considering each possible value of $D_{20}$:

- If $D_{20} = 1$, $D_{30}$ can be 29 values (2-30).
- If $D_{20} = 2$, $D_{30}$ can be 28 values (3-30).
...
- If $D_{20} = 20$, $D_{30}$ can be 10 values (21-30).

So the number of favorable outcomes is 
$$29+...+10 = (29+...+1)  - (9+...+1) = \frac{(29)(30)}{2} - \frac{(9)(10)}{2} = \frac{1}{2} (870-90)= 390$$

Thus, our answer is $\frac{390}{600} = \boxed{\frac{13}{20}}$

</details>

# 60 Heads 1
### Question
A fair coin is flipped 100 times. Using the Central Limit Theorem without continuity correction, estimate the probability that at least 60 heads are obtained. The answer is in form $\Phi(a)$ for some $a$, where $\Phi$ is the standard normal distribution CDF. Find $a$.

<details>
  <summary>Answer</summary>

### Answer

Let $n=100$ and $p=0.5$. According to the CLT, for a larger number of trials, the binomial distribution can be approximated by a normal distribution with mean $\mu = np = 100 \times 0.5 = 50$ and standard deviation $\sigma = \sqrt{np(1-p)} = \sqrt{100 \times 0.5 \times 0.5} = 5$. 

We need to find the probability $P(X \geq 60)$. Using the normal approximation, this is equivalent to finding $P(Z \geq 2)$ where $Z$ is a standard normal variable.
Hence,
$$
P(X \geq 60) \approx P(Z \geq 2) = 1 - \Phi(2)
$$

$\Phi(z) = 1- \Phi(-2)$ is true because the standard normal distribution's CDF is symmetric around the mean (0), which means that the probability of a standard normal variable being less than or equal to $z$ is equal to the probability of it being greater than or equal to $-z$. Thus,
$$ 1 - \Phi(2) = \Phi(-2) \Rightarrow a =\boxed{-2}$$
</details>

# Longer Piece 1
### Question
Suppose that you have a loaf of bread that is 1 meter in length. You slice the piece of bread at a uniformly random point along its length, creating 2 slices. Find the expected length (in meters) of the longest piece of bread. 

<details>
  <summary>Answer</summary>

### Answer

Let’s denote the position where the bread is sliced by $X$ , where $X$ is a random variable uniformly distributed between 0 and 1 (representing the position along the 1 meter length of the bread). To find the expected length of the longer piece, we first determine the length of the longer piece given a slice at $X$: $L = \max(X, 1 - X)$ The expected value of L can be computed by integrating over the entire range of X from 0 to 1. Specifically: $E[L] = \int^1_0 \text{max}(X,1-X)dX$. Since this function is not differentiable at $X = 0.5$, we split up this integral since from $0 \leq X \leq 0.5$, the longer piece is $1-X$. For $0.5 < X \leq 1$, the longer piece is X.
Thus, 
$$
E[L] = \int^{0.5}_{0} (1-X)dX + \int^{1}_{0.5} XdX = \left[ X - \frac{X^2}{2}\right]^{0.5}_0 + \left[\frac{X^2}{2}\right]^1_{0.5} 
$$
$$= \left(0.5 - \frac{0.5^2}{2}\right) + \left(\frac{1^2}{2} - \frac{0.5^2}{2}\right) = 0.375+0.375 = \boxed{0.75} 
$$
</details>

# Coin Swap 1
### Question
Alice and Bob play a game with a coin with probability $0 < p \leq 1$ of heads per flip.  Alice starts with the coin and flips it. The game stops when someone flips a heads. If either player flip a tails, they pass the coin to the other player for them to flip. Find the probability that Alice is the winner when $p = \frac{1}{3}$
<details>
  <summary>Answer</summary>

### Answer

There are two approaches to this question:

One involves abusing the fact that the game is recursive and really only has two game states. Let $P(A)$ denote the probability that the player that flips first wins, which happens to be what we are looking for since Alice flips first. At the start of the game, Bob's win probability is $1-P(A)$. This is the first game state. The initial flip by Alice leads to 2 possible outcomes: Alice can win outright with a probability of $p$, or Alice passes the coin to Bob with the probability of $1-p$. If this outcome occurs, the game enters the second state, where Bob flips. Now, using some mental gymnastics, we can observe that this second game state is identical to the first game state except the fact that the person with the initial flip is Bob. Thus Bob's win probability becomes $P(A)$ and Alice's win probability is $1-P(A)$, since Bob is the initial coin flipper. Until the game concludes, the game will always alternate between these two states. Alice wins if she flips heads initially, which happens with probability p, and if Alice flips tails (probability $1-p$), then Bob flips the coin, and the probability that Alice wins from this state where Bob starts is $1-P(A)$.
Thus, $P(A) = p+(1-p)(1-P(A))$, and solving this equation with $p=\frac{1}{3}$ gives us $\boxed{\frac{3}{5}}$


 Another solution involves summing up the probabilities of the scenarios where Alice wins. The quickest scenario is  when Alice rolls heads outright, and the probability this occurs is $\frac{1}{3}$. The next fastest scenario is when Alice rolls tails, Bobs rolls tails, and Alice rolls heads. The probability this occurs is $\frac{2}{3} \cdot \frac{2}{3} \cdot \frac{1}{3}$. The next fastest scenario occurs when outcomes goes tails, tails, tails, tails, heads, and the probability of this is $\frac{2}{3} \cdot \frac{2}{3} \cdot \frac{2}{3} \cdot \frac{2}{3} \cdot  \frac{1}{3}$... The sum of these probabilities can be expressed as so 
$$
\frac{1}{3}\sum_{n=0}^{\infty}\left(\frac{4}{9}\right)^n = \frac{1}{3} \left(\frac{1}{1-\frac{4}{9}}\right) = \boxed{\frac{3}{5}}
$$
</details>


# LinkedIn Networking 1
### Question
You message $N$  quant researchers on LinkedIn in an attempt to get a referral. You created a personalized message for each, but forgot which one corresponds to each person. Therefore, you send each of them a random message, with each message being used exactly once. Find the expected number of quant researchers that receive the message you intended for them when $N = 20$
<details>
  <summary>Answer</summary>

### Answer
Let $X_i$ be a random variable indicating whether the $i$=th researcher receives the correct message. $X_i=1$ if the researcher receives the correct message, and $X_i=0$ otherwise. The probability that the $i$-th researcher receives the correct message is $\frac{1}{N}$. Therefore, the expected value of $X_i$ is: $E[X_i] = \frac{1}{N}$. Let $X$ be the total number of researchers who receive the correct message. Then, $X = X_1+X_2+...+X_N$ BY the linearity of expectation, $E[X] = E[X_1+X_2+...+X_N] = E[X_1]+E[X_2]+...+E[X_N]= N \cdot \frac{1}{N} = \boxed{1}$
</details>

# Consecutive 1
### Question
On average, how many times must a fair standard die be rolled to obtain two consecutive 1s.
<details>
  <summary>Answer</summary>

### Answer

</details>

We can use Markov chains to solve this problem. This game has three game states. Let State 0 denote the start of the game, State 1 denote the state  where a 1 has been rolled, and State 2 where two ones have been rolled, which is the absorption state. Let $E_0$ be the expected number of rolls from the start to get two consecutive 1's, and $E_1$ be the expected number of rolls after having rolled one 1 (but not two consecutive 1’s). From state 0, there is a $\frac{1}{6}$ probability on entering state 1, and a $\frac{5}{6}$ probability of staying in state 0. From state 1, there is a $\frac{1}{6}$ probability to enter the absorption state, and a $\frac{5}{6}$ opportunity of returning to state 0. Thus, we can setup the following equations:
$$E_2=0$$
$$E_0=1+\frac{1}{6}E_1+\frac{5}{6}E_0$$
$$E_1=1+\frac{1}{6}E_2+\frac{5}{6}E_0$$

We can solve this system of equations for $E_0$, which would give us $E_0 = \boxed{42}$



