"""
LSTM Numerical Example - Implemented From Scratch
==================================================
Based on: LSTM_numerical_example_pdf.pdf & Lecture_3_pdf.pdf
          (A. Prof. Noha El-Attar)

Task   : Predict the next value in the sequence [1, 2, 3] -> ~= 4
Model  : LSTM with input_dim=1, hidden_dim=1
Output : Linear layer  y_hat = W_y * h_t + b_y
"""

import math


# ---------------------------------------------------------------------------
# Helper activation functions
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """sigma(x) = 1 / (1 + e^{-x})  -- squashes output to (0, 1)"""
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x: float) -> float:
    """tanh(x) -- squashes output to (-1, 1)"""
    return math.tanh(x)


# ---------------------------------------------------------------------------
# LSTM Parameters  (all scalars: input_dim=1, hidden_dim=1)
#
# Each gate:  gate = sigma / tanh ( W_x * x_t  +  W_h * h_{t-1}  +  b )
#
# Forget gate  (f)  ->  how much of the old cell state to KEEP
# Input  gate  (i)  ->  how much NEW information to WRITE
# Candidate    (c~) ->  candidate values to potentially add to cell state
# Output gate  (o)  ->  how much of the cell state to READ / output
# ---------------------------------------------------------------------------

# Forget gate weights & bias
W_xf = 0.45;  W_hf = 0.25;  b_f = 0.0

# Input gate weights & bias
W_xi = 0.95;  W_hi = 0.60;  b_i = 0.0

# Candidate cell-state weights & bias
W_xc = 0.45;  W_hc = 0.25;  b_c = 0.0

# Output gate weights & bias
W_xo = 0.60;  W_ho = 0.45;  b_o = 0.0

# Prediction (output) layer
W_y = 4.05;   b_y = 0.5

# ---------------------------------------------------------------------------
# Input sequence & initial states
# ---------------------------------------------------------------------------

sequence = [1, 2, 3]   # input time steps  (predict the next value ~= 4)
h = 0.0                # h_{t-1}  initial hidden state
C = 0.0                # C_{t-1}  initial cell state

# ---------------------------------------------------------------------------
# Forward pass -- verbose, step-by-step (mirrors the PDF)
# ---------------------------------------------------------------------------

print("=" * 62)
print("  LSTM Forward Pass -- Step-by-Step")
print("=" * 62)

for t, x in enumerate(sequence, start=1):
    print(f"\n{'--' * 31}")
    print(f"  Time step t = {t},  x_t = {x}")
    print(f"{'--' * 31}")

    # 1. Forget gate  F_t = sigma(x_t * W_xf + h_{t-1} * W_hf + b_f)
    f_pre = x * W_xf + h * W_hf + b_f
    f = sigmoid(f_pre)
    print(f"  Forget gate : F_{t} = sigma({x} * {W_xf} + {h:.4f} * {W_hf} + {b_f})")
    print(f"              = sigma({f_pre:.4f}) = {f:.4f}")

    # 2. Input gate  I_t = sigma(x_t * W_xi + h_{t-1} * W_hi + b_i)
    i_pre = x * W_xi + h * W_hi + b_i
    i = sigmoid(i_pre)
    print(f"  Input  gate : I_{t} = sigma({x} * {W_xi} + {h:.4f} * {W_hi} + {b_i})")
    print(f"              = sigma({i_pre:.4f}) = {i:.4f}")

    # 3. Candidate  C~_t = tanh(x_t * W_xc + h_{t-1} * W_hc + b_c)
    c_pre = x * W_xc + h * W_hc + b_c
    c_tilde = tanh(c_pre)
    print(f"  Candidate   : C~_{t} = tanh({x} * {W_xc} + {h:.4f} * {W_hc} + {b_c})")
    print(f"              = tanh({c_pre:.4f}) = {c_tilde:.4f}")

    # 4. Cell state  C_t = F_t * C_{t-1} + I_t * C~_t
    C_new = f * C + i * c_tilde
    print(f"  Cell state  : C_{t} = F_{t} * C_{{t-1}} + I_{t} * C~_{t}")
    print(f"              = {f:.4f} * {C:.4f} + {i:.4f} * {c_tilde:.4f}")
    print(f"              = {C_new:.4f}")

    # 5. Output gate  O_t = sigma(x_t * W_xo + h_{t-1} * W_ho + b_o)
    o_pre = x * W_xo + h * W_ho + b_o
    o = sigmoid(o_pre)
    print(f"  Output gate : O_{t} = sigma({x} * {W_xo} + {h:.4f} * {W_ho} + {b_o})")
    print(f"              = sigma({o_pre:.4f}) = {o:.4f}")

    # 6. Hidden state  H_t = O_t * tanh(C_t)
    h_new = o * tanh(C_new)
    print(f"  Hidden state: H_{t} = O_{t} * tanh(C_{t})")
    print(f"              = {o:.4f} * tanh({C_new:.4f})")
    print(f"              = {h_new:.4f}")

    # Advance states for the next time step
    C = C_new
    h = h_new

# Prediction  y_hat = W_y * h_3 + b_y
y_hat = W_y * h + b_y

print(f"\n{'=' * 62}")
print(f"  Final hidden state : h_3   = {h:.4f}")
print(f"  Prediction         : y_hat = {W_y} * h_3 + {b_y}")
print(f"                             = {W_y} * {h:.4f} + {b_y}")
print(f"                             = {y_hat:.4f}  ~=  {round(y_hat, 1)}")
print(f"  Expected next value in [1, 2, 3, ?]:  ~= 4  (model -> ~3.8)  OK")
print(f"{'=' * 62}\n")


# ---------------------------------------------------------------------------
# LSTMCell class -- clean, reusable implementation
# ---------------------------------------------------------------------------

class LSTMCell:
    """
    A single-layer, scalar LSTM cell (input_dim=1, hidden_dim=1).

    Parameters are fixed at construction time.
    Call ``forward(x, h_prev, C_prev)`` at each time step to get the
    updated hidden state and cell state.
    """

    def __init__(
        self,
        W_xf: float, W_hf: float, b_f: float,
        W_xi: float, W_hi: float, b_i: float,
        W_xc: float, W_hc: float, b_c: float,
        W_xo: float, W_ho: float, b_o: float,
        W_y:  float, b_y:  float,
    ) -> None:
        # Forget gate
        self.W_xf, self.W_hf, self.b_f = W_xf, W_hf, b_f
        # Input gate
        self.W_xi, self.W_hi, self.b_i = W_xi, W_hi, b_i
        # Candidate
        self.W_xc, self.W_hc, self.b_c = W_xc, W_hc, b_c
        # Output gate
        self.W_xo, self.W_ho, self.b_o = W_xo, W_ho, b_o
        # Prediction layer
        self.W_y, self.b_y = W_y, b_y

    def forward(self, x: float, h_prev: float, C_prev: float) -> tuple:
        """
        One forward step of the LSTM cell.

        Args:
            x      : scalar input at the current time step
            h_prev : hidden state from the previous time step
            C_prev : cell state from the previous time step

        Returns:
            (h_t, C_t) -- updated hidden state and cell state
        """
        f = sigmoid(x * self.W_xf + h_prev * self.W_hf + self.b_f)
        i = sigmoid(x * self.W_xi + h_prev * self.W_hi + self.b_i)
        c_tilde = tanh(x * self.W_xc + h_prev * self.W_hc + self.b_c)
        C_t = f * C_prev + i * c_tilde
        o = sigmoid(x * self.W_xo + h_prev * self.W_ho + self.b_o)
        h_t = o * tanh(C_t)
        return h_t, C_t

    def predict(self, h_t: float) -> float:
        """Linear output layer: y_hat = W_y * h_t + b_y"""
        return self.W_y * h_t + self.b_y

    def run_sequence(self, xs: list) -> float:
        """
        Run the full sequence and return the final prediction.

        Args:
            xs : list of scalar inputs

        Returns:
            y_hat -- scalar prediction after the last time step
        """
        h, C = 0.0, 0.0
        for x in xs:
            h, C = self.forward(x, h, C)
        return self.predict(h)


# ---------------------------------------------------------------------------
# Demo: LSTMCell class producing the same result
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cell = LSTMCell(
        W_xf=0.45, W_hf=0.25, b_f=0.0,
        W_xi=0.95, W_hi=0.60, b_i=0.0,
        W_xc=0.45, W_hc=0.25, b_c=0.0,
        W_xo=0.60, W_ho=0.45, b_o=0.0,
        W_y=4.05,  b_y=0.5,
    )

    y_hat_class = cell.run_sequence([1, 2, 3])
    print(
        f"LSTMCell.run_sequence([1, 2, 3]) -> y_hat = {y_hat_class:.4f}"
        f"  ~=  {round(y_hat_class, 1)}"  # adjacent f-strings are auto-concatenated
    )
