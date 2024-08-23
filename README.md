# Snake Game AI Bot

## Project Overview

This project focuses on creating an AI bot capable of playing the classic Snake game using reinforcement learning techniques, specifically Q-learning. The AI is trained to maximize its score by collecting food while avoiding obstacles.

## Objective

The goal of the project is to develop an AI bot that can successfully play the Snake game by utilizing reinforcement learning methods. The bot is trained to understand the game's environment and make decisions that lead to higher scores.

## AI Mechanisms

### Rewards:

- **+1000 points** for collecting food.
- **+3 points** for moving closer to the food.

### Penalties:

- **-200 points** for hitting an obstacle.
- **-3 points** for moving away from the food.
- **-100 points** if the bot fails to collect food within a certain time frame (getting lost).

### AI State Representation:

The state of the game is represented by:

- **Food position**: Encoded as [up, down, left, right] with values of 0 or 1.
  - Example: If the food is on the right side of the snake: `[0, 0, 0, 1]`.
- **Proximity to food**: Whether the snake is getting closer to the food (1 for yes, 0 for no).
- **Snake direction**: Right (1), Left (-1), Up/Down (0).
- **Obstacle position**: Encoded similarly to the food position as [up, down, left, right].

## Training Methodology

The AI bot was trained under various scenarios:

1. **Training with a snake length of 3** (with and without the ability to grow).
2. **Training with a snake length of 11** (without the ability to grow).
3. **Training for different snake lengths separately** and combining the results. The AI uses a separate knowledge dictionary for each length, switching to a different one when a reward is collected.

### Example Training Scenarios

#### Training with a Snake Length of 3 (Limit 10)
![Training with Snake Length 3 (Limit 10)](./length3limit10/ezgif.com-gif-maker%20(4).gif)

#### Training with a Snake Length of 3 (Limit 3)
![Training with Snake Length 3 (Limit 3)](./length3Limit3/ezgif.com-gif-maker%20(3).gif)

#### Training with a Snake Length of 11
![Training with Snake Length 11](./lengthOnly11/ezgif.com-gif-maker%20(1).gif)

### Insights:

- Training the snake as per game rules is challenging due to changing game conditions with increasing length.
- Training the snake separately for each length showed promising results.
- If growth is ignored, the model can play indefinitely.

## Dependencies

The following libraries were used to develop the Snake AI bot:

- **PyGame Learning Environment**
- **numpy**
- **pillow**
- **pygame**
