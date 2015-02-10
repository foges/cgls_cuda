
%% Test 3
n = 5;

val_real = [  1, -1, -3, -2,  5,  4,  6,  4, -4,  2,  7,  8, -5];
val_imag = [  0,  2, -1,  0,  0,  0,  0,  0, -9,  8,  1, -1,  1];

row_ptr  = [0, 3, 5, 8, 11, 13];
col_ind  = [0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4];

b_real   = [ -2, -1,  0,  1,  2]';
b_imag   = [  2,  1,  0, -1, -2]';

% From github.com/foges/bmt
A = csr2sparse(val_real + 1i * val_imag, row_ptr, col_ind, n);

shift = 0;
x_star = (A' * A + shift * eye(5)) \ (A' * (b_real + 1i * b_imag));
