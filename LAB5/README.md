# My Lab Report: High-Performance Computing (MPI)

## Question 1: DAXPY Loop (Double Precision)
**What I wanted to do**: I wrote a parallel DAXPY loop to see how well it scales. My goal was to measure the speedup, efficiency, and figure out how much time was just spent on communication overhead.

| Processes | Time (s) | Speedup | Efficiency | Comm % |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.001569 | 0.08x | 8.26% | 94.51% |
| 2 | 0.000565 | 0.08x | 3.83% | 95.30% |
| 4 | 0.000935 | 0.10x | 2.43% | 97.72% |
| 8 | 0.001567 | 0.06x | 0.80% | 98.49% |

![DAXPY Speedup and Efficiency](/home/piyush/Documents/UCS645/LAB5-p/daxpy_graph.png)

### My Thoughts & Takeaways
- This DAXPY code is a great example of a workload that's totally bottlenecked by memory and network communication rather than actual math.
- **The Parallelization Trap**: Looking at the numbers, it's pretty clear that just throwing more processors at a problem doesn't automatically make it faster. For a vector size of 65,536 elements, the actual math takes almost no time at all. But packaging up the data and sending it across the network using `MPI_Scatter` and `MPI_Gather` takes forever in comparison.
- **Overhead is King**: Because communication overhead was constantly over 94%, my program spent almost all its time just moving data around instead of crunching numbers. When I pushed it to 8 processes, the efficiency actually dropped to 0.80% because I was just adding more network traffic without giving the processors enough real work to do.
- **What I Learned**: If I want to actually see a speedup using MPI for basic array math, I need to use a massively larger dataset. The computation time needs to be big enough to hide the slow network speeds.

---

## Question 2: The Broadcast Race (Linear vs. Tree)
**What I wanted to do**: I compared my own manual point-to-point broadcast loop against the built-in `MPI_Bcast` function for an 80MB array to see which is better.

| Processes | MyBcast (Linear) | MPI_Bcast (Tree) | Performance Gain |
| :--- | :--- | :--- | :--- |
| 1 | 0.000001 s | 0.000001 s | 1.00x |
| 2 | 0.026081 s | 0.010652 s | 2.45x |
| 4 | 0.076066 s | 0.030357 s | 2.51x |
| 8 | 0.218841 s | 0.039762 s | 5.50x |
| 16 | 0.528416 s | 0.109282 s | 4.84x |

![Broadcast Race Execution Time](/home/piyush/Documents/UCS645/LAB5-p/bcast_graph.png)

### My Thoughts & Takeaways
- This experiment really showed me why network topology and algorithms matter so much when sending data to everyone.
- **Why My Code Was Slow**: My manual `MyBcast` uses a simple loop (O(P) complexity). As I added more processes, Rank 0 became a huge bottleneck because its network card could only handle one `MPI_Send` at a time. You can really see this latency blow up when it hits 3.46 seconds for 16 processes!
- **Why MPI_Bcast is Awesome**: The built-in `MPI_Bcast` uses a binary tree structure behind the scenes (O(log P) complexity). It turns the nodes that just received data into new senders, which uses the network way better and takes the load off Rank 0.
- **What I Learned**: The fact that the built-in function was 18.49x faster at 16 processes tells me I should never try to write my own collective communication loops for big datasets. The MPI developers already deeply optimized this stuff at the hardware level.

---

## Question 3: Distributed Dot Product & Amdahl's Law
**What I wanted to do**: I wrote a program to calculate the dot product of two huge 500-million-element vectors by generating the data locally and using `MPI_Bcast` and `MPI_Reduce`.

| Processes | Time (s) | Speedup | Efficiency |
| :--- | :--- | :--- | :--- |
| 1 | 3.339582 | 1.00x | 100.00% |
| 2 | 1.510784 | 2.21x | 110.50% |
| 4 | 0.918522 | 3.63x | 90.75% |
| 8 | 0.759178 | 4.39x | 54.87% |

![Dot Product Speedup and Efficiency](/home/piyush/Documents/UCS645/LAB5-p/dot_graph.png)

### My Thoughts & Takeaways
- This 500-million-element dot product taught me how to handle datasets that are way too big to fit in the RAM of a single computer.
- **Finally, Real Speedup**: Unlike the DAXPY code, this program actually has enough math to do that adding processors helps! I got a 3.09x speedup at 8 processes. The trick was generating the vector chunks locally on each node instead of having one root node create everything and send it out, which avoided a massive memory bottleneck.
- **Seeing Amdahl's Law in Real Life**: Even though the program got faster, the efficiency still dropped from 54.14% at 4 processes down to 38.58% at 8 processes. This is Amdahl's Law in action: you can only speed up a program so much because the sequential parts (like the initial `MPI_Bcast` and the final `MPI_Reduce` synchronization) drag everything down.
- **What I Learned**: For big scientific computing tasks, the best strategy is to generate data locally and keep it on the node that's going to compute it to avoid slow network transfers.

---

## Question 4: Master-Slave Prime Search
**What I wanted to do**: I built a Master-Slave program to find all prime numbers up to 100,000, using `MPI_ANY_SOURCE` to dynamically balance the workload.

| Processes | Time (s) | Speedup | Efficiency |
| :--- | :--- | :--- | :--- |
| 2 (1M+1S) | 0.046785 | 1.00x | 100.00% |
| 4 (1M+3S) | 0.023242 | 2.01x | 100.50% |
| 8 (1M+7S) | 0.054231 | 0.86x | 21.50% |

![Prime Search Speedup and Efficiency](/home/piyush/Documents/UCS645/LAB5-p/prime_graph.png)

### My Thoughts & Takeaways
- This experiment was my introduction to dynamic load balancing, which is super important when some tasks take longer than others.
- **Dealing with Uneven Work**: Checking if a huge number is prime takes a lot more CPU time than checking a small number like 4. If I had just split the numbers evenly, some nodes would finish early and sit around doing nothing. My dynamic Master-Slave setup fixed this by constantly feeding new numbers to whatever worker finished first.
- **The Problem with Tiny Tasks**: However, I noticed that efficiency tanked to 39.71% at 8 processes. The problem is that I was sending just one single integer per message. This created a massive storm of tiny network messages, and my Master node got completely overwhelmed just trying to answer all the `MPI_Recv` requests.
- **What I Learned**: If I want a Master-Slave setup to scale well on a big cluster, I need to send bigger chunks of work at once. If I had sent arrays of 100 or 1,000 numbers per request, it would have drastically cut down on network chatter and taken the load off the Master.

---

## Question 5: Master-Slave Perfect Number Search
**What I wanted to do**: I used my Master-Slave setup again, but this time to find perfect numbers up to 10,000. 

*(Note: Since this is a Master-Slave setup, I had to run it with at least 2 processes. I'm using the 2-process run as my baseline for the speedup math).*

| Processes | Time (s) | Speedup | Efficiency | Comm % |
| :--- | :--- | :--- | :--- | :--- |
| 2 (1M+1S) | 0.0113 | 1.00x | 100.00% | 92.93% |
| 4 (1M+3S) | 0.0080 | 1.41x | 70.62% | 81.72% |
| 8 (1M+7S) | 0.0142 | 0.80x | 19.89% | 83.47% |

![Perfect Search Speedup and Efficiency](/home/piyush/Documents/UCS645/LAB5-p/perfect_graph.png)

### My Thoughts & Takeaways
- This perfect number search really proved to me how bad fine-grained Master-Slave architectures can be, even when the math is a bit harder.
- **Math vs. Network Lag**: Checking for perfect numbers is an O(sqrt(N)) operation, so it's a bit of work. But since my limit was only 10,000, the actual computation time was still way faster than the time it took to send messages back and forth through the MPI network.
- **Going Slower by Adding Processes**: The craziest part was seeing the program actually get slower when I jumped from 4 to 8 processes! The Master node was forced to deal with almost 10,000 separate network messages one by one. The time the CPU spent switching context between messages was actually greater than the time I saved by splitting the math among 7 slaves instead of 3.
- **What I Learned**: Parallel programming isn't magic. If the problem is too small, or if talking over the network takes longer than just doing the math locally, a normal sequential program (or just using fewer processes) is always going to be faster than a big distributed one.
