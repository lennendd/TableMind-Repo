<div align="center">
<h1>
TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning
</h1>
</div>

## 📖 Abstract
**TableMind introduces the study of autonomous table reasoning with large language models. We propose a two-stage fine-tuning framework that combines supervised trajectory learning with reinforcement optimization via RAPO, a rank-aware strategy for improving reasoning accuracy. Our agent performs multi-turn tool invocation, executes code in a secure sandbox for precise numerical analysis, and leverages planning and reflection for adaptive strategies.**

## 🌟 Overview
<div align="center">
<img src="./image/Overview.png" width="90%"/>
<p><em></em></p>
</div>

## Key Features

- **Multi-turn Tool Calling**: End-to-end reinforcement learning on complete interaction trajectories, allowing agents to learn from sequences of actions
- **Multi-tool Coordination**: Train agents to effectively coordinate and use multiple tools together to solve complex tasks
- **Process Rewards**: Assign rewards for each tool call based on its effectiveness, balanced with outcome rewards through normalization
- **Custom Tools and Environments**: Compatible with mainstream LLM tool calling formats, making it easy to extend with your own tools and scenarios
- **Multiple RL Algorithms**: Supports diverse reinforcement learning approaches including `PPO`, `GRPO`, and `REINFORCE++`
- **Multi-modal Support**: Compatible with vision-language models (VLMs) and multi-modal reinforcement learning

<p align="center"><img src="./image/Framework.png" width="800px" alt="" /></p>

## Extending Agent with Your Own Tools and Environments

Agent-R1 provides a flexible architecture for creating custom tools and tool environments to suit various agent applications. Our framework is built on two key abstractions:

1. **BaseTool**: Individual tools that agents can use to interact with external systems
2. **BaseToolEnv**: Tool environments that define the state transition function for agent-tool interactions