function z=calculateZ(pre_A, W, b)

b=repmat(b, 1, size(pre_A, 2));
z=W*pre_A+b;

end