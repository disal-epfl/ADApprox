# MIT License
#
# Copyright (c) 2025 Nicolaj Bösel-Schmid
# Contact: nicolaj.schmid@epfl.ch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


class SchurComplement():
    def __init__(
        self,
    ):
        pass

    def inverse(
        self,
        M: np.ndarray,
    ):
        """
        Calculate the inverse of the matrix M using the Schur complement.
        M is assumed to be an 4x4 or 3x3 block matrix where each block is a N
        dimensional diagonal matrix that is collapsed into a vector.
        Args:
            M (np.ndarray): The matrix to calculate the inverse of (4, 4, N) or (3, 3, N).
        Returns:
            M_inv (np.ndarray): The inverse matrix (4, 4, N) or (3, 3, N).
        """
        M = np.copy(M)
        assert M.shape[0] == M.shape[1]
        assert M.shape[0] == 4 or M.shape[0] == 3

        # Define block matrices
        A = M[:2, :2] # (2, 2, N)
        B = M[:2, 2:] # (2, 2, N) or (2, 1, N)
        C = M[2:, :2] # (2, 2, N) or (1, 2, N)
        D = M[2:, 2:] # (2, 2, N) or (1, 1, N)

        # TODO: implement for 3x3 block matrices, adapt _mul

        # Calculate the Schur complement
        A_inv = self._inv_2by2(A)
        S = D - self._mul(C, self._mul(A_inv, B))
        if M.shape[0] == 3:
            S_inv = self._inv_1by1(S)
        else:
            S_inv = self._inv_2by2(S)

        # Calculate the inverse of the matrix
        M_inv = np.zeros_like(M)
        M_inv[:2, :2] = A_inv + self._mul(A_inv, self._mul(B, self._mul(S_inv, self._mul(C, A_inv))))
        M_inv[:2, 2:] = - self._mul(A_inv, self._mul(B, S_inv))
        M_inv[2:, :2] = - self._mul(S_inv, self._mul(C, A_inv))
        M_inv[2:, 2:] = S_inv
        return M_inv

    def _inv_1by1(
        self,
        M: np.ndarray,
    ):
        """
        Calculate the inverse of a 1x1 diagonal block matrix.
        Args:
            M (np.ndarray): The matrix to calculate the inverse of (1, 1, N).
        Returns:
            np.ndarray: The inverse of the matrix (1, 1, N).
        """
        return 1 / M

    def _inv_2by2(
        self,
        M: np.ndarray,
    ):
        """
        Calculate the Schur complement of a matrix having a 2x2 block structure.
        The blocks are assumed to be 2x2 diagonal matrices.
        Args:
            M (np.ndarray): The matrix to calculate the inverse of (2, 2, N).
        Returns:
            np.ndarray: The inverse of the Schur complement.
        """
        A_inv = 1 / M[0, 0]
        B = M[0, 1]
        C = M[1, 0]
        D = M[1, 1]

        S_inv = 1 / (D - C * A_inv * B)
        return np.array([
            [A_inv + A_inv * B * S_inv * C * A_inv, -A_inv * B * S_inv],
            [-S_inv * C * A_inv,                    S_inv],
        ])

    def _mul(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ):
        """
        Multiply two matrices. The matrices have a 2x2 or 1x2 block structure.
        Each block is a diagonal matrices having N entries.
        Args:
            A (np.ndarray): The first matrix (2, 2, N) or (1, 2, N).
            B (np.ndarray): The second matrix (2, 2, N) or (2, 1, N).
        Returns:
            C (np.ndarray): The product of the two matrices (2, 2, N), (1, 2, N) or (2, 1, N).
        """
        return np.einsum('ilk, ljk -> ijk', A, B)
        # C = np.zeros_like(A)
        # C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
        # C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
        # C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
        # C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
        # return C


        
    # def inverse2(
    #     self,
    #     M: np.ndarray,
    # ):
    #     """
    #     Calculate the Schur complement of a matrix. The matrix
    #     M is assumed to be an DNxDN block matrix where each block 
    #     Mij is a DxD diagonal matrix. D is the number of dimensions
    #     and N is the number of blocks.
    #     Args:
    #         M (np.ndarray): The matrix to calculate the inverse of (D, D, N).
    #     Returns:
    #         np.ndarray: The inverse of the Schur complement.
    #     """
    #     M = np.copy(M)
    #     D = M.shape[0]
    #     assert M.shape[0] == M.shape[1]
    #     assert np.log2(D) % 1 == 0
    #     assert D == 4

    #     A_inv = self._inv_diag(
    #             A_inv = 1 / M[0, 0],
    #             B = M[0, 1],
    #             C = M[1, 0],
    #             D = M[1, 1],
    #         )
    #     A_inv = np.concatenate([
    #         np.concatenate([np.diag(A_inv[0, 0]), np.diag(A_inv[0, 1])], axis=1),
    #         np.concatenate([np.diag(A_inv[1, 0]), np.diag(A_inv[1, 1])], axis=1),
    #     ], axis=0)
    #     B = np.concatenate([
    #         np.concatenate([np.diag(M[0, 2]), np.diag(M[0, 3])], axis=1),
    #         np.concatenate([np.diag(M[1, 2]), np.diag(M[1, 3])], axis=1),
    #     ], axis=0)
    #     C = np.concatenate([
    #         np.concatenate([np.diag(M[2, 0]), np.diag(M[2, 1])], axis=1),
    #         np.concatenate([np.diag(M[3, 0]), np.diag(M[3, 1])], axis=1),
    #     ], axis=0)
    #     D = np.concatenate([
    #         np.concatenate([np.diag(M[2, 2]), np.diag(M[2, 3])], axis=1),
    #         np.concatenate([np.diag(M[3, 2]), np.diag(M[3, 3])], axis=1),
    #     ], axis=0)

    #     return self._inv(
    #         A_inv = A_inv,
    #         B = B,
    #         C = C,
    #         D = D,
    #     )

    # def _inv(
    #     self,
    #     A_inv: np.ndarray,
    #     B: np.ndarray,
    #     C: np.ndarray,
    #     D: np.ndarray,
    # ):
    #     """
    #     Calculate the Schur complement of a matrix having a 2x2 block structure.
    #     The blocks are assumed to be 2x2 diagonal matrices.
    #     Args:
    #         A_inv (np.ndarray): The inverse of the top-left block (N, N).
    #         B (np.ndarray): The top-right block (N, N).
    #         C (np.ndarray): The bottom-left block (N, N).
    #         D (np.ndarray): The bottom-right block (N, N).
    #     Returns:
    #         np.ndarray: The inverse of the Schur complement.
    #     """
    #     S_inv = np.linalg.inv(D - C @ A_inv @ B)

    #     return np.concatenate([
    #         np.concatenate([A_inv + A_inv @ B @ S_inv @ C @ A_inv, -A_inv @ B @ S_inv], axis=1),
    #         np.concatenate([-S_inv @ C @ A_inv, S_inv], axis=1),
    #     ], axis=0)

    # def _inv_diag(
    #     self,
    #     A_inv: np.ndarray,
    #     B: np.ndarray,
    #     C: np.ndarray,
    #     D: np.ndarray,
    # ):
    #     """
    #     Calculate the Schur complement of a matrix having a 2x2 block structure.
    #     The blocks are assumed to be 2x2 diagonal matrices.
    #     Args:
    #         A_inv (np.ndarray): The inverse of the top-left block (N).
    #         B (np.ndarray): The top-right block (N).
    #         C (np.ndarray): The bottom-left block (N).
    #         D (np.ndarray): The bottom-right block (N).
    #     Returns:
    #         np.ndarray: The inverse of the Schur complement.
    #     """
    #     S_inv = 1 / (D - C * A_inv * B)

    #     return np.array([
    #         [A_inv + A_inv * B * S_inv * C * A_inv, -A_inv * B * S_inv],
    #         [-S_inv * C * A_inv,                    S_inv],
    #     ])
        