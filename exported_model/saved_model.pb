??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( ?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ۛ
t
input/PlaceholderPlaceholder*'
_output_shapes
:?????????	*
dtype0*
shape:?????????	
?
3layer_1/weights1/Initializer/truncated_normal/shapeConst*#
_class
loc:@layer_1/weights1*
_output_shapes
:*
dtype0*
valueB"	   K   
?
2layer_1/weights1/Initializer/truncated_normal/meanConst*#
_class
loc:@layer_1/weights1*
_output_shapes
: *
dtype0*
valueB
 *    
?
4layer_1/weights1/Initializer/truncated_normal/stddevConst*#
_class
loc:@layer_1/weights1*
_output_shapes
: *
dtype0*
valueB
 * ?3>
?
=layer_1/weights1/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3layer_1/weights1/Initializer/truncated_normal/shape*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:	K*
dtype0
?
1layer_1/weights1/Initializer/truncated_normal/mulMul=layer_1/weights1/Initializer/truncated_normal/TruncatedNormal4layer_1/weights1/Initializer/truncated_normal/stddev*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:	K
?
-layer_1/weights1/Initializer/truncated_normalAdd1layer_1/weights1/Initializer/truncated_normal/mul2layer_1/weights1/Initializer/truncated_normal/mean*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:	K
?
layer_1/weights1VarHandleOp*#
_class
loc:@layer_1/weights1*
_output_shapes
: *
dtype0*
shape
:	K*!
shared_namelayer_1/weights1
q
1layer_1/weights1/IsInitialized/VarIsInitializedOpVarIsInitializedOplayer_1/weights1*
_output_shapes
: 
y
layer_1/weights1/AssignAssignVariableOplayer_1/weights1-layer_1/weights1/Initializer/truncated_normal*
dtype0
u
$layer_1/weights1/Read/ReadVariableOpReadVariableOplayer_1/weights1*
_output_shapes

:	K*
dtype0
?
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
_output_shapes
:K*
dtype0*
valueBK*    
?
layer_1/biases1VarHandleOp*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0*
shape:K* 
shared_namelayer_1/biases1
o
0layer_1/biases1/IsInitialized/VarIsInitializedOpVarIsInitializedOplayer_1/biases1*
_output_shapes
: 
k
layer_1/biases1/AssignAssignVariableOplayer_1/biases1!layer_1/biases1/Initializer/zeros*
dtype0
o
#layer_1/biases1/Read/ReadVariableOpReadVariableOplayer_1/biases1*
_output_shapes
:K*
dtype0
n
layer_1/MatMul/ReadVariableOpReadVariableOplayer_1/weights1*
_output_shapes

:	K*
dtype0
|
layer_1/MatMulMatMulinput/Placeholderlayer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????K
f
layer_1/add/ReadVariableOpReadVariableOplayer_1/biases1*
_output_shapes
:K*
dtype0
r
layer_1/addAddV2layer_1/MatMullayer_1/add/ReadVariableOp*
T0*'
_output_shapes
:?????????K
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:?????????K
?
3layer_2/weights2/Initializer/truncated_normal/shapeConst*#
_class
loc:@layer_2/weights2*
_output_shapes
:*
dtype0*
valueB"K   d   
?
2layer_2/weights2/Initializer/truncated_normal/meanConst*#
_class
loc:@layer_2/weights2*
_output_shapes
: *
dtype0*
valueB
 *    
?
4layer_2/weights2/Initializer/truncated_normal/stddevConst*#
_class
loc:@layer_2/weights2*
_output_shapes
: *
dtype0*
valueB
 *???=
?
=layer_2/weights2/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3layer_2/weights2/Initializer/truncated_normal/shape*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:Kd*
dtype0
?
1layer_2/weights2/Initializer/truncated_normal/mulMul=layer_2/weights2/Initializer/truncated_normal/TruncatedNormal4layer_2/weights2/Initializer/truncated_normal/stddev*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:Kd
?
-layer_2/weights2/Initializer/truncated_normalAdd1layer_2/weights2/Initializer/truncated_normal/mul2layer_2/weights2/Initializer/truncated_normal/mean*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:Kd
?
layer_2/weights2VarHandleOp*#
_class
loc:@layer_2/weights2*
_output_shapes
: *
dtype0*
shape
:Kd*!
shared_namelayer_2/weights2
q
1layer_2/weights2/IsInitialized/VarIsInitializedOpVarIsInitializedOplayer_2/weights2*
_output_shapes
: 
y
layer_2/weights2/AssignAssignVariableOplayer_2/weights2-layer_2/weights2/Initializer/truncated_normal*
dtype0
u
$layer_2/weights2/Read/ReadVariableOpReadVariableOplayer_2/weights2*
_output_shapes

:Kd*
dtype0
?
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
_output_shapes
:d*
dtype0*
valueBd*    
?
layer_2/biases2VarHandleOp*"
_class
loc:@layer_2/biases2*
_output_shapes
: *
dtype0*
shape:d* 
shared_namelayer_2/biases2
o
0layer_2/biases2/IsInitialized/VarIsInitializedOpVarIsInitializedOplayer_2/biases2*
_output_shapes
: 
k
layer_2/biases2/AssignAssignVariableOplayer_2/biases2!layer_2/biases2/Initializer/zeros*
dtype0
o
#layer_2/biases2/Read/ReadVariableOpReadVariableOplayer_2/biases2*
_output_shapes
:d*
dtype0
n
layer_2/MatMul/ReadVariableOpReadVariableOplayer_2/weights2*
_output_shapes

:Kd*
dtype0
w
layer_2/MatMulMatMullayer_1/Relulayer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d
f
layer_2/add/ReadVariableOpReadVariableOplayer_2/biases2*
_output_shapes
:d*
dtype0
r
layer_2/addAddV2layer_2/MatMullayer_2/add/ReadVariableOp*
T0*'
_output_shapes
:?????????d
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:?????????d
?
3layer_3/weights3/Initializer/truncated_normal/shapeConst*#
_class
loc:@layer_3/weights3*
_output_shapes
:*
dtype0*
valueB"d   2   
?
2layer_3/weights3/Initializer/truncated_normal/meanConst*#
_class
loc:@layer_3/weights3*
_output_shapes
: *
dtype0*
valueB
 *    
?
4layer_3/weights3/Initializer/truncated_normal/stddevConst*#
_class
loc:@layer_3/weights3*
_output_shapes
: *
dtype0*
valueB
 *l>
?
=layer_3/weights3/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3layer_3/weights3/Initializer/truncated_normal/shape*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:d2*
dtype0
?
1layer_3/weights3/Initializer/truncated_normal/mulMul=layer_3/weights3/Initializer/truncated_normal/TruncatedNormal4layer_3/weights3/Initializer/truncated_normal/stddev*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:d2
?
-layer_3/weights3/Initializer/truncated_normalAdd1layer_3/weights3/Initializer/truncated_normal/mul2layer_3/weights3/Initializer/truncated_normal/mean*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:d2
?
layer_3/weights3VarHandleOp*#
_class
loc:@layer_3/weights3*
_output_shapes
: *
dtype0*
shape
:d2*!
shared_namelayer_3/weights3
q
1layer_3/weights3/IsInitialized/VarIsInitializedOpVarIsInitializedOplayer_3/weights3*
_output_shapes
: 
y
layer_3/weights3/AssignAssignVariableOplayer_3/weights3-layer_3/weights3/Initializer/truncated_normal*
dtype0
u
$layer_3/weights3/Read/ReadVariableOpReadVariableOplayer_3/weights3*
_output_shapes

:d2*
dtype0
?
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
_output_shapes
:2*
dtype0*
valueB2*    
?
layer_3/biases3VarHandleOp*"
_class
loc:@layer_3/biases3*
_output_shapes
: *
dtype0*
shape:2* 
shared_namelayer_3/biases3
o
0layer_3/biases3/IsInitialized/VarIsInitializedOpVarIsInitializedOplayer_3/biases3*
_output_shapes
: 
k
layer_3/biases3/AssignAssignVariableOplayer_3/biases3!layer_3/biases3/Initializer/zeros*
dtype0
o
#layer_3/biases3/Read/ReadVariableOpReadVariableOplayer_3/biases3*
_output_shapes
:2*
dtype0
n
layer_3/MatMul/ReadVariableOpReadVariableOplayer_3/weights3*
_output_shapes

:d2*
dtype0
w
layer_3/MatMulMatMullayer_2/Relulayer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
f
layer_3/add/ReadVariableOpReadVariableOplayer_3/biases3*
_output_shapes
:2*
dtype0
r
layer_3/addAddV2layer_3/MatMullayer_3/add/ReadVariableOp*
T0*'
_output_shapes
:?????????2
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:?????????2
?
2output/weights4/Initializer/truncated_normal/shapeConst*"
_class
loc:@output/weights4*
_output_shapes
:*
dtype0*
valueB"2      
?
1output/weights4/Initializer/truncated_normal/meanConst*"
_class
loc:@output/weights4*
_output_shapes
: *
dtype0*
valueB
 *    
?
3output/weights4/Initializer/truncated_normal/stddevConst*"
_class
loc:@output/weights4*
_output_shapes
: *
dtype0*
valueB
 *L?f>
?
<output/weights4/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2output/weights4/Initializer/truncated_normal/shape*
T0*"
_class
loc:@output/weights4*
_output_shapes

:2*
dtype0
?
0output/weights4/Initializer/truncated_normal/mulMul<output/weights4/Initializer/truncated_normal/TruncatedNormal3output/weights4/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@output/weights4*
_output_shapes

:2
?
,output/weights4/Initializer/truncated_normalAdd0output/weights4/Initializer/truncated_normal/mul1output/weights4/Initializer/truncated_normal/mean*
T0*"
_class
loc:@output/weights4*
_output_shapes

:2
?
output/weights4VarHandleOp*"
_class
loc:@output/weights4*
_output_shapes
: *
dtype0*
shape
:2* 
shared_nameoutput/weights4
o
0output/weights4/IsInitialized/VarIsInitializedOpVarIsInitializedOpoutput/weights4*
_output_shapes
: 
v
output/weights4/AssignAssignVariableOpoutput/weights4,output/weights4/Initializer/truncated_normal*
dtype0
s
#output/weights4/Read/ReadVariableOpReadVariableOpoutput/weights4*
_output_shapes

:2*
dtype0
?
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
_output_shapes
:*
dtype0*
valueB*    
?
output/biases4VarHandleOp*!
_class
loc:@output/biases4*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/biases4
m
/output/biases4/IsInitialized/VarIsInitializedOpVarIsInitializedOpoutput/biases4*
_output_shapes
: 
h
output/biases4/AssignAssignVariableOpoutput/biases4 output/biases4/Initializer/zeros*
dtype0
m
"output/biases4/Read/ReadVariableOpReadVariableOpoutput/biases4*
_output_shapes
:*
dtype0
l
output/MatMul/ReadVariableOpReadVariableOpoutput/weights4*
_output_shapes

:2*
dtype0
u
output/MatMulMatMullayer_3/Reluoutput/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
d
output/add/ReadVariableOpReadVariableOpoutput/biases4*
_output_shapes
:*
dtype0
o

output/addAddV2output/MatMuloutput/add/ReadVariableOp*
T0*'
_output_shapes
:?????????
Q
output/ReluRelu
output/add*
T0*'
_output_shapes
:?????????
s
cost/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
cost/SquaredDifferenceSquaredDifferenceoutput/Relucost/Placeholder*
T0*'
_output_shapes
:?????????
[

cost/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
V
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
d
train/gradients/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
z
train/gradients/grad_ys_0Filltrain/gradients/Shapetrain/gradients/grad_ys_0/Const*
T0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
?
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/grad_ys_0,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
_output_shapes

:
j
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
_output_shapes
:
?
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*
T0*'
_output_shapes
:?????????
l
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
n
$train/gradients/cost/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: 
p
&train/gradients/cost/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: 
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
?
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
?
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
T0*
_output_shapes
: 
?
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
?
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:?????????
?
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @
?
/train/gradients/cost/SquaredDifference_grad/MulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:?????????
?
/train/gradients/cost/SquaredDifference_grad/subSuboutput/Relucost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:?????????
?
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/Mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:?????????
l
1train/gradients/cost/SquaredDifference_grad/ShapeShapeoutput/Relu*
T0*
_output_shapes
:
s
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
_output_shapes
:
?
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
?
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*'
_output_shapes
:?????????
?
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*'
_output_shapes
:?????????
?
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:?????????
?
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
?
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:?????????
?
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:?????????
?
)train/gradients/output/Relu_grad/ReluGradReluGradDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyoutput/Relu*
T0*'
_output_shapes
:?????????
b
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
_output_shapes
:
p
'train/gradients/output/add_grad/Shape_1Shapeoutput/add/ReadVariableOp*
T0*
_output_shapes
:
?
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
#train/gradients/output/add_grad/SumSum)train/gradients/output/Relu_grad/ReluGrad5train/gradients/output/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
?
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*'
_output_shapes
:?????????
?
%train/gradients/output/add_grad/Sum_1Sum)train/gradients/output/Relu_grad/ReluGrad7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
T0*
_output_shapes
:
?
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
?
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:?????????
?
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
?
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2*
transpose_b(
?
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
T0*
_output_shapes

:2*
transpose_a(
?
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
?
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:?????????2
?
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:2
?
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:?????????2
d
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Shapelayer_3/add/ReadVariableOp*
T0*
_output_shapes
:
?
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
?
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*'
_output_shapes
:?????????2
?
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
_output_shapes
:2
?
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
?
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:?????????2
?
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:2
?
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d*
transpose_b(
?
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:d2*
transpose_a(
?
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
?
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:?????????d
?
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes

:d2
?
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:?????????d
d
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Shapelayer_2/add/ReadVariableOp*
T0*
_output_shapes
:
?
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
?
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*'
_output_shapes
:?????????d
?
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
T0*
_output_shapes
:d
?
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
?
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:?????????d
?
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:d
?
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????K*
transpose_b(
?
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:Kd*
transpose_a(
?
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
?
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:?????????K
?
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:Kd
?
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:?????????K
d
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Shapelayer_1/add/ReadVariableOp*
T0*
_output_shapes
:
?
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
?
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*'
_output_shapes
:?????????K
?
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
_output_shapes
:K
?
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
?
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:?????????K
?
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:K
?
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	*
transpose_b(
?
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:	K*
transpose_a(
?
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
?
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:?????????	
?
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:	K
?
+train/beta1_power/Initializer/initial_valueConst*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0*
valueB
 *fff?
?
train/beta1_powerVarHandleOp*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0*
shape: *"
shared_nametrain/beta1_power
?
2train/beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/beta1_power*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
y
train/beta1_power/AssignAssignVariableOptrain/beta1_power+train/beta1_power/Initializer/initial_value*
dtype0
?
%train/beta1_power/Read/ReadVariableOpReadVariableOptrain/beta1_power*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0
?
+train/beta2_power/Initializer/initial_valueConst*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0*
valueB
 *w??
?
train/beta2_powerVarHandleOp*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0*
shape: *"
shared_nametrain/beta2_power
?
2train/beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/beta2_power*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
y
train/beta2_power/AssignAssignVariableOptrain/beta2_power+train/beta2_power/Initializer/initial_value*
dtype0
?
%train/beta2_power/Read/ReadVariableOpReadVariableOptrain/beta2_power*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0
?
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
_output_shapes

:	K*
dtype0*
valueB	K*    
?
train/layer_1/weights1/AdamVarHandleOp*#
_class
loc:@layer_1/weights1*
_output_shapes
: *
dtype0*
shape
:	K*,
shared_nametrain/layer_1/weights1/Adam
?
<train/layer_1/weights1/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_1/weights1/Adam*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
?
"train/layer_1/weights1/Adam/AssignAssignVariableOptrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
dtype0
?
/train/layer_1/weights1/Adam/Read/ReadVariableOpReadVariableOptrain/layer_1/weights1/Adam*#
_class
loc:@layer_1/weights1*
_output_shapes

:	K*
dtype0
?
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
_output_shapes

:	K*
dtype0*
valueB	K*    
?
train/layer_1/weights1/Adam_1VarHandleOp*#
_class
loc:@layer_1/weights1*
_output_shapes
: *
dtype0*
shape
:	K*.
shared_nametrain/layer_1/weights1/Adam_1
?
>train/layer_1/weights1/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_1/weights1/Adam_1*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
?
$train/layer_1/weights1/Adam_1/AssignAssignVariableOptrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
dtype0
?
1train/layer_1/weights1/Adam_1/Read/ReadVariableOpReadVariableOptrain/layer_1/weights1/Adam_1*#
_class
loc:@layer_1/weights1*
_output_shapes

:	K*
dtype0
?
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
_output_shapes
:K*
dtype0*
valueBK*    
?
train/layer_1/biases1/AdamVarHandleOp*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0*
shape:K*+
shared_nametrain/layer_1/biases1/Adam
?
;train/layer_1/biases1/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_1/biases1/Adam*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
?
!train/layer_1/biases1/Adam/AssignAssignVariableOptrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
dtype0
?
.train/layer_1/biases1/Adam/Read/ReadVariableOpReadVariableOptrain/layer_1/biases1/Adam*"
_class
loc:@layer_1/biases1*
_output_shapes
:K*
dtype0
?
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
_output_shapes
:K*
dtype0*
valueBK*    
?
train/layer_1/biases1/Adam_1VarHandleOp*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0*
shape:K*-
shared_nametrain/layer_1/biases1/Adam_1
?
=train/layer_1/biases1/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_1/biases1/Adam_1*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
?
#train/layer_1/biases1/Adam_1/AssignAssignVariableOptrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
dtype0
?
0train/layer_1/biases1/Adam_1/Read/ReadVariableOpReadVariableOptrain/layer_1/biases1/Adam_1*"
_class
loc:@layer_1/biases1*
_output_shapes
:K*
dtype0
?
=train/layer_2/weights2/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@layer_2/weights2*
_output_shapes
:*
dtype0*
valueB"K   d   
?
3train/layer_2/weights2/Adam/Initializer/zeros/ConstConst*#
_class
loc:@layer_2/weights2*
_output_shapes
: *
dtype0*
valueB
 *    
?
-train/layer_2/weights2/Adam/Initializer/zerosFill=train/layer_2/weights2/Adam/Initializer/zeros/shape_as_tensor3train/layer_2/weights2/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:Kd
?
train/layer_2/weights2/AdamVarHandleOp*#
_class
loc:@layer_2/weights2*
_output_shapes
: *
dtype0*
shape
:Kd*,
shared_nametrain/layer_2/weights2/Adam
?
<train/layer_2/weights2/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_2/weights2/Adam*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
?
"train/layer_2/weights2/Adam/AssignAssignVariableOptrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
dtype0
?
/train/layer_2/weights2/Adam/Read/ReadVariableOpReadVariableOptrain/layer_2/weights2/Adam*#
_class
loc:@layer_2/weights2*
_output_shapes

:Kd*
dtype0
?
?train/layer_2/weights2/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@layer_2/weights2*
_output_shapes
:*
dtype0*
valueB"K   d   
?
5train/layer_2/weights2/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@layer_2/weights2*
_output_shapes
: *
dtype0*
valueB
 *    
?
/train/layer_2/weights2/Adam_1/Initializer/zerosFill?train/layer_2/weights2/Adam_1/Initializer/zeros/shape_as_tensor5train/layer_2/weights2/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:Kd
?
train/layer_2/weights2/Adam_1VarHandleOp*#
_class
loc:@layer_2/weights2*
_output_shapes
: *
dtype0*
shape
:Kd*.
shared_nametrain/layer_2/weights2/Adam_1
?
>train/layer_2/weights2/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_2/weights2/Adam_1*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
?
$train/layer_2/weights2/Adam_1/AssignAssignVariableOptrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
dtype0
?
1train/layer_2/weights2/Adam_1/Read/ReadVariableOpReadVariableOptrain/layer_2/weights2/Adam_1*#
_class
loc:@layer_2/weights2*
_output_shapes

:Kd*
dtype0
?
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
_output_shapes
:d*
dtype0*
valueBd*    
?
train/layer_2/biases2/AdamVarHandleOp*"
_class
loc:@layer_2/biases2*
_output_shapes
: *
dtype0*
shape:d*+
shared_nametrain/layer_2/biases2/Adam
?
;train/layer_2/biases2/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_2/biases2/Adam*"
_class
loc:@layer_2/biases2*
_output_shapes
: 
?
!train/layer_2/biases2/Adam/AssignAssignVariableOptrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
dtype0
?
.train/layer_2/biases2/Adam/Read/ReadVariableOpReadVariableOptrain/layer_2/biases2/Adam*"
_class
loc:@layer_2/biases2*
_output_shapes
:d*
dtype0
?
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
_output_shapes
:d*
dtype0*
valueBd*    
?
train/layer_2/biases2/Adam_1VarHandleOp*"
_class
loc:@layer_2/biases2*
_output_shapes
: *
dtype0*
shape:d*-
shared_nametrain/layer_2/biases2/Adam_1
?
=train/layer_2/biases2/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_2/biases2/Adam_1*"
_class
loc:@layer_2/biases2*
_output_shapes
: 
?
#train/layer_2/biases2/Adam_1/AssignAssignVariableOptrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
dtype0
?
0train/layer_2/biases2/Adam_1/Read/ReadVariableOpReadVariableOptrain/layer_2/biases2/Adam_1*"
_class
loc:@layer_2/biases2*
_output_shapes
:d*
dtype0
?
=train/layer_3/weights3/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@layer_3/weights3*
_output_shapes
:*
dtype0*
valueB"d   2   
?
3train/layer_3/weights3/Adam/Initializer/zeros/ConstConst*#
_class
loc:@layer_3/weights3*
_output_shapes
: *
dtype0*
valueB
 *    
?
-train/layer_3/weights3/Adam/Initializer/zerosFill=train/layer_3/weights3/Adam/Initializer/zeros/shape_as_tensor3train/layer_3/weights3/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:d2
?
train/layer_3/weights3/AdamVarHandleOp*#
_class
loc:@layer_3/weights3*
_output_shapes
: *
dtype0*
shape
:d2*,
shared_nametrain/layer_3/weights3/Adam
?
<train/layer_3/weights3/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_3/weights3/Adam*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
?
"train/layer_3/weights3/Adam/AssignAssignVariableOptrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
dtype0
?
/train/layer_3/weights3/Adam/Read/ReadVariableOpReadVariableOptrain/layer_3/weights3/Adam*#
_class
loc:@layer_3/weights3*
_output_shapes

:d2*
dtype0
?
?train/layer_3/weights3/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@layer_3/weights3*
_output_shapes
:*
dtype0*
valueB"d   2   
?
5train/layer_3/weights3/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@layer_3/weights3*
_output_shapes
: *
dtype0*
valueB
 *    
?
/train/layer_3/weights3/Adam_1/Initializer/zerosFill?train/layer_3/weights3/Adam_1/Initializer/zeros/shape_as_tensor5train/layer_3/weights3/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:d2
?
train/layer_3/weights3/Adam_1VarHandleOp*#
_class
loc:@layer_3/weights3*
_output_shapes
: *
dtype0*
shape
:d2*.
shared_nametrain/layer_3/weights3/Adam_1
?
>train/layer_3/weights3/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_3/weights3/Adam_1*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
?
$train/layer_3/weights3/Adam_1/AssignAssignVariableOptrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
dtype0
?
1train/layer_3/weights3/Adam_1/Read/ReadVariableOpReadVariableOptrain/layer_3/weights3/Adam_1*#
_class
loc:@layer_3/weights3*
_output_shapes

:d2*
dtype0
?
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
_output_shapes
:2*
dtype0*
valueB2*    
?
train/layer_3/biases3/AdamVarHandleOp*"
_class
loc:@layer_3/biases3*
_output_shapes
: *
dtype0*
shape:2*+
shared_nametrain/layer_3/biases3/Adam
?
;train/layer_3/biases3/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_3/biases3/Adam*"
_class
loc:@layer_3/biases3*
_output_shapes
: 
?
!train/layer_3/biases3/Adam/AssignAssignVariableOptrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
dtype0
?
.train/layer_3/biases3/Adam/Read/ReadVariableOpReadVariableOptrain/layer_3/biases3/Adam*"
_class
loc:@layer_3/biases3*
_output_shapes
:2*
dtype0
?
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
_output_shapes
:2*
dtype0*
valueB2*    
?
train/layer_3/biases3/Adam_1VarHandleOp*"
_class
loc:@layer_3/biases3*
_output_shapes
: *
dtype0*
shape:2*-
shared_nametrain/layer_3/biases3/Adam_1
?
=train/layer_3/biases3/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/layer_3/biases3/Adam_1*"
_class
loc:@layer_3/biases3*
_output_shapes
: 
?
#train/layer_3/biases3/Adam_1/AssignAssignVariableOptrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
dtype0
?
0train/layer_3/biases3/Adam_1/Read/ReadVariableOpReadVariableOptrain/layer_3/biases3/Adam_1*"
_class
loc:@layer_3/biases3*
_output_shapes
:2*
dtype0
?
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
_output_shapes

:2*
dtype0*
valueB2*    
?
train/output/weights4/AdamVarHandleOp*"
_class
loc:@output/weights4*
_output_shapes
: *
dtype0*
shape
:2*+
shared_nametrain/output/weights4/Adam
?
;train/output/weights4/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/output/weights4/Adam*"
_class
loc:@output/weights4*
_output_shapes
: 
?
!train/output/weights4/Adam/AssignAssignVariableOptrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
dtype0
?
.train/output/weights4/Adam/Read/ReadVariableOpReadVariableOptrain/output/weights4/Adam*"
_class
loc:@output/weights4*
_output_shapes

:2*
dtype0
?
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
_output_shapes

:2*
dtype0*
valueB2*    
?
train/output/weights4/Adam_1VarHandleOp*"
_class
loc:@output/weights4*
_output_shapes
: *
dtype0*
shape
:2*-
shared_nametrain/output/weights4/Adam_1
?
=train/output/weights4/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/output/weights4/Adam_1*"
_class
loc:@output/weights4*
_output_shapes
: 
?
#train/output/weights4/Adam_1/AssignAssignVariableOptrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
dtype0
?
0train/output/weights4/Adam_1/Read/ReadVariableOpReadVariableOptrain/output/weights4/Adam_1*"
_class
loc:@output/weights4*
_output_shapes

:2*
dtype0
?
+train/output/biases4/Adam/Initializer/zerosConst*!
_class
loc:@output/biases4*
_output_shapes
:*
dtype0*
valueB*    
?
train/output/biases4/AdamVarHandleOp*!
_class
loc:@output/biases4*
_output_shapes
: *
dtype0*
shape:**
shared_nametrain/output/biases4/Adam
?
:train/output/biases4/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/output/biases4/Adam*!
_class
loc:@output/biases4*
_output_shapes
: 
?
 train/output/biases4/Adam/AssignAssignVariableOptrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
dtype0
?
-train/output/biases4/Adam/Read/ReadVariableOpReadVariableOptrain/output/biases4/Adam*!
_class
loc:@output/biases4*
_output_shapes
:*
dtype0
?
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
_output_shapes
:*
dtype0*
valueB*    
?
train/output/biases4/Adam_1VarHandleOp*!
_class
loc:@output/biases4*
_output_shapes
: *
dtype0*
shape:*,
shared_nametrain/output/biases4/Adam_1
?
<train/output/biases4/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptrain/output/biases4/Adam_1*!
_class
loc:@output/biases4*
_output_shapes
: 
?
"train/output/biases4/Adam_1/AssignAssignVariableOptrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
dtype0
?
/train/output/biases4/Adam_1/Read/ReadVariableOpReadVariableOptrain/output/biases4/Adam_1*!
_class
loc:@output/biases4*
_output_shapes
:*
dtype0
]
train/Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
U
train/Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
U
train/Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w??
W
train/Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2
?
Ctrain/Adam/update_layer_1/weights1/ResourceApplyAdam/ReadVariableOpReadVariableOptrain/beta1_power*
_output_shapes
: *
dtype0
?
Etrain/Adam/update_layer_1/weights1/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptrain/beta2_power*
_output_shapes
: *
dtype0
?
4train/Adam/update_layer_1/weights1/ResourceApplyAdamResourceApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1Ctrain/Adam/update_layer_1/weights1/ResourceApplyAdam/ReadVariableOpEtrain/Adam/update_layer_1/weights1/ResourceApplyAdam/ReadVariableOp_1train/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_1/weights1
?
Btrain/Adam/update_layer_1/biases1/ResourceApplyAdam/ReadVariableOpReadVariableOptrain/beta1_power*
_output_shapes
: *
dtype0
?
Dtrain/Adam/update_layer_1/biases1/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptrain/beta2_power*
_output_shapes
: *
dtype0
?
3train/Adam/update_layer_1/biases1/ResourceApplyAdamResourceApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1Btrain/Adam/update_layer_1/biases1/ResourceApplyAdam/ReadVariableOpDtrain/Adam/update_layer_1/biases1/ResourceApplyAdam/ReadVariableOp_1train/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_1/biases1
?
Ctrain/Adam/update_layer_2/weights2/ResourceApplyAdam/ReadVariableOpReadVariableOptrain/beta1_power*
_output_shapes
: *
dtype0
?
Etrain/Adam/update_layer_2/weights2/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptrain/beta2_power*
_output_shapes
: *
dtype0
?
4train/Adam/update_layer_2/weights2/ResourceApplyAdamResourceApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1Ctrain/Adam/update_layer_2/weights2/ResourceApplyAdam/ReadVariableOpEtrain/Adam/update_layer_2/weights2/ResourceApplyAdam/ReadVariableOp_1train/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_2/weights2
?
Btrain/Adam/update_layer_2/biases2/ResourceApplyAdam/ReadVariableOpReadVariableOptrain/beta1_power*
_output_shapes
: *
dtype0
?
Dtrain/Adam/update_layer_2/biases2/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptrain/beta2_power*
_output_shapes
: *
dtype0
?
3train/Adam/update_layer_2/biases2/ResourceApplyAdamResourceApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1Btrain/Adam/update_layer_2/biases2/ResourceApplyAdam/ReadVariableOpDtrain/Adam/update_layer_2/biases2/ResourceApplyAdam/ReadVariableOp_1train/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_2/biases2
?
Ctrain/Adam/update_layer_3/weights3/ResourceApplyAdam/ReadVariableOpReadVariableOptrain/beta1_power*
_output_shapes
: *
dtype0
?
Etrain/Adam/update_layer_3/weights3/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptrain/beta2_power*
_output_shapes
: *
dtype0
?
4train/Adam/update_layer_3/weights3/ResourceApplyAdamResourceApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1Ctrain/Adam/update_layer_3/weights3/ResourceApplyAdam/ReadVariableOpEtrain/Adam/update_layer_3/weights3/ResourceApplyAdam/ReadVariableOp_1train/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_3/weights3
?
Btrain/Adam/update_layer_3/biases3/ResourceApplyAdam/ReadVariableOpReadVariableOptrain/beta1_power*
_output_shapes
: *
dtype0
?
Dtrain/Adam/update_layer_3/biases3/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptrain/beta2_power*
_output_shapes
: *
dtype0
?
3train/Adam/update_layer_3/biases3/ResourceApplyAdamResourceApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1Btrain/Adam/update_layer_3/biases3/ResourceApplyAdam/ReadVariableOpDtrain/Adam/update_layer_3/biases3/ResourceApplyAdam/ReadVariableOp_1train/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_3/biases3
?
Btrain/Adam/update_output/weights4/ResourceApplyAdam/ReadVariableOpReadVariableOptrain/beta1_power*
_output_shapes
: *
dtype0
?
Dtrain/Adam/update_output/weights4/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptrain/beta2_power*
_output_shapes
: *
dtype0
?
3train/Adam/update_output/weights4/ResourceApplyAdamResourceApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1Btrain/Adam/update_output/weights4/ResourceApplyAdam/ReadVariableOpDtrain/Adam/update_output/weights4/ResourceApplyAdam/ReadVariableOp_1train/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
T0*"
_class
loc:@output/weights4
?
Atrain/Adam/update_output/biases4/ResourceApplyAdam/ReadVariableOpReadVariableOptrain/beta1_power*
_output_shapes
: *
dtype0
?
Ctrain/Adam/update_output/biases4/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptrain/beta2_power*
_output_shapes
: *
dtype0
?
2train/Adam/update_output/biases4/ResourceApplyAdamResourceApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1Atrain/Adam/update_output/biases4/ResourceApplyAdam/ReadVariableOpCtrain/Adam/update_output/biases4/ResourceApplyAdam/ReadVariableOp_1train/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
T0*!
_class
loc:@output/biases4
?
train/Adam/ReadVariableOpReadVariableOptrain/beta1_power4^train/Adam/update_layer_1/biases1/ResourceApplyAdam5^train/Adam/update_layer_1/weights1/ResourceApplyAdam4^train/Adam/update_layer_2/biases2/ResourceApplyAdam5^train/Adam/update_layer_2/weights2/ResourceApplyAdam4^train/Adam/update_layer_3/biases3/ResourceApplyAdam5^train/Adam/update_layer_3/weights3/ResourceApplyAdam3^train/Adam/update_output/biases4/ResourceApplyAdam4^train/Adam/update_output/weights4/ResourceApplyAdam*
_output_shapes
: *
dtype0
?
train/Adam/mulMultrain/Adam/ReadVariableOptrain/Adam/beta1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
?
train/Adam/AssignVariableOpAssignVariableOptrain/beta1_powertrain/Adam/mul*"
_class
loc:@layer_1/biases1*
dtype0
?
train/Adam/ReadVariableOp_1ReadVariableOptrain/beta1_power^train/Adam/AssignVariableOp4^train/Adam/update_layer_1/biases1/ResourceApplyAdam5^train/Adam/update_layer_1/weights1/ResourceApplyAdam4^train/Adam/update_layer_2/biases2/ResourceApplyAdam5^train/Adam/update_layer_2/weights2/ResourceApplyAdam4^train/Adam/update_layer_3/biases3/ResourceApplyAdam5^train/Adam/update_layer_3/weights3/ResourceApplyAdam3^train/Adam/update_output/biases4/ResourceApplyAdam4^train/Adam/update_output/weights4/ResourceApplyAdam*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0
?
train/Adam/ReadVariableOp_2ReadVariableOptrain/beta2_power4^train/Adam/update_layer_1/biases1/ResourceApplyAdam5^train/Adam/update_layer_1/weights1/ResourceApplyAdam4^train/Adam/update_layer_2/biases2/ResourceApplyAdam5^train/Adam/update_layer_2/weights2/ResourceApplyAdam4^train/Adam/update_layer_3/biases3/ResourceApplyAdam5^train/Adam/update_layer_3/weights3/ResourceApplyAdam3^train/Adam/update_output/biases4/ResourceApplyAdam4^train/Adam/update_output/weights4/ResourceApplyAdam*
_output_shapes
: *
dtype0
?
train/Adam/mul_1Multrain/Adam/ReadVariableOp_2train/Adam/beta2*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
?
train/Adam/AssignVariableOp_1AssignVariableOptrain/beta2_powertrain/Adam/mul_1*"
_class
loc:@layer_1/biases1*
dtype0
?
train/Adam/ReadVariableOp_3ReadVariableOptrain/beta2_power^train/Adam/AssignVariableOp_14^train/Adam/update_layer_1/biases1/ResourceApplyAdam5^train/Adam/update_layer_1/weights1/ResourceApplyAdam4^train/Adam/update_layer_2/biases2/ResourceApplyAdam5^train/Adam/update_layer_2/weights2/ResourceApplyAdam4^train/Adam/update_layer_3/biases3/ResourceApplyAdam5^train/Adam/update_layer_3/weights3/ResourceApplyAdam3^train/Adam/update_output/biases4/ResourceApplyAdam4^train/Adam/update_output/weights4/ResourceApplyAdam*"
_class
loc:@layer_1/biases1*
_output_shapes
: *
dtype0
?

train/AdamNoOp^train/Adam/AssignVariableOp^train/Adam/AssignVariableOp_14^train/Adam/update_layer_1/biases1/ResourceApplyAdam5^train/Adam/update_layer_1/weights1/ResourceApplyAdam4^train/Adam/update_layer_2/biases2/ResourceApplyAdam5^train/Adam/update_layer_2/weights2/ResourceApplyAdam4^train/Adam/update_layer_3/biases3/ResourceApplyAdam5^train/Adam/update_layer_3/weights3/ResourceApplyAdam3^train/Adam/update_output/biases4/ResourceApplyAdam4^train/Adam/update_output/weights4/ResourceApplyAdam
n
logging/current_cost/tagsConst*
_output_shapes
: *
dtype0*%
valueB Blogging/current_cost
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
s
logging/predicted_value/tagConst*
_output_shapes
: *
dtype0*(
valueB Blogging/predicted_value
m
logging/predicted_valueHistogramSummarylogging/predicted_value/tagoutput/Relu*
_output_shapes
: 
z
logging/Merge/MergeSummaryMergeSummarylogging/current_costlogging/predicted_value*
N*
_output_shapes
: 
?
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign

init_all_tablesNoOp
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp\part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
?

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices#layer_1/biases1/Read/ReadVariableOp$layer_1/weights1/Read/ReadVariableOp#layer_2/biases2/Read/ReadVariableOp$layer_2/weights2/Read/ReadVariableOp#layer_3/biases3/Read/ReadVariableOp$layer_3/weights3/Read/ReadVariableOp"output/biases4/Read/ReadVariableOp#output/weights4/Read/ReadVariableOp%train/beta1_power/Read/ReadVariableOp%train/beta2_power/Read/ReadVariableOp.train/layer_1/biases1/Adam/Read/ReadVariableOp0train/layer_1/biases1/Adam_1/Read/ReadVariableOp/train/layer_1/weights1/Adam/Read/ReadVariableOp1train/layer_1/weights1/Adam_1/Read/ReadVariableOp.train/layer_2/biases2/Adam/Read/ReadVariableOp0train/layer_2/biases2/Adam_1/Read/ReadVariableOp/train/layer_2/weights2/Adam/Read/ReadVariableOp1train/layer_2/weights2/Adam_1/Read/ReadVariableOp.train/layer_3/biases3/Adam/Read/ReadVariableOp0train/layer_3/biases3/Adam_1/Read/ReadVariableOp/train/layer_3/weights3/Adam/Read/ReadVariableOp1train/layer_3/weights3/Adam_1/Read/ReadVariableOp-train/output/biases4/Adam/Read/ReadVariableOp/train/output/biases4/Adam_1/Read/ReadVariableOp.train/output/weights4/Adam/Read/ReadVariableOp0train/output/weights4/Adam_1/Read/ReadVariableOp"/device:CPU:0*(
dtypes
2
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
X
save/AssignVariableOpAssignVariableOplayer_1/biases1save/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
[
save/AssignVariableOp_1AssignVariableOplayer_1/weights1save/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
Z
save/AssignVariableOp_2AssignVariableOplayer_2/biases2save/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
[
save/AssignVariableOp_3AssignVariableOplayer_2/weights2save/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
Z
save/AssignVariableOp_4AssignVariableOplayer_3/biases3save/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
[
save/AssignVariableOp_5AssignVariableOplayer_3/weights3save/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
Y
save/AssignVariableOp_6AssignVariableOpoutput/biases4save/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
Z
save/AssignVariableOp_7AssignVariableOpoutput/weights4save/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
\
save/AssignVariableOp_8AssignVariableOptrain/beta1_powersave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
]
save/AssignVariableOp_9AssignVariableOptrain/beta2_powersave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
g
save/AssignVariableOp_10AssignVariableOptrain/layer_1/biases1/Adamsave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
i
save/AssignVariableOp_11AssignVariableOptrain/layer_1/biases1/Adam_1save/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
h
save/AssignVariableOp_12AssignVariableOptrain/layer_1/weights1/Adamsave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
j
save/AssignVariableOp_13AssignVariableOptrain/layer_1/weights1/Adam_1save/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
g
save/AssignVariableOp_14AssignVariableOptrain/layer_2/biases2/Adamsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
i
save/AssignVariableOp_15AssignVariableOptrain/layer_2/biases2/Adam_1save/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
h
save/AssignVariableOp_16AssignVariableOptrain/layer_2/weights2/Adamsave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
j
save/AssignVariableOp_17AssignVariableOptrain/layer_2/weights2/Adam_1save/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
g
save/AssignVariableOp_18AssignVariableOptrain/layer_3/biases3/Adamsave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
i
save/AssignVariableOp_19AssignVariableOptrain/layer_3/biases3/Adam_1save/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
h
save/AssignVariableOp_20AssignVariableOptrain/layer_3/weights3/Adamsave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
j
save/AssignVariableOp_21AssignVariableOptrain/layer_3/weights3/Adam_1save/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
f
save/AssignVariableOp_22AssignVariableOptrain/output/biases4/Adamsave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
h
save/AssignVariableOp_23AssignVariableOptrain/output/biases4/Adam_1save/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
g
save/AssignVariableOp_24AssignVariableOptrain/output/weights4/Adamsave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
i
save/AssignVariableOp_25AssignVariableOptrain/output/weights4/Adam_1save/Identity_26*
dtype0
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"*
saved_model_main_op

init_all_tables"B
	summaries5
3
logging/current_cost:0
logging/predicted_value:0"
train_op


train/Adam"?
trainable_variables??
?
layer_1/weights1:0layer_1/weights1/Assign&layer_1/weights1/Read/ReadVariableOp:0(2/layer_1/weights1/Initializer/truncated_normal:08
{
layer_1/biases1:0layer_1/biases1/Assign%layer_1/biases1/Read/ReadVariableOp:0(2#layer_1/biases1/Initializer/zeros:08
?
layer_2/weights2:0layer_2/weights2/Assign&layer_2/weights2/Read/ReadVariableOp:0(2/layer_2/weights2/Initializer/truncated_normal:08
{
layer_2/biases2:0layer_2/biases2/Assign%layer_2/biases2/Read/ReadVariableOp:0(2#layer_2/biases2/Initializer/zeros:08
?
layer_3/weights3:0layer_3/weights3/Assign&layer_3/weights3/Read/ReadVariableOp:0(2/layer_3/weights3/Initializer/truncated_normal:08
{
layer_3/biases3:0layer_3/biases3/Assign%layer_3/biases3/Read/ReadVariableOp:0(2#layer_3/biases3/Initializer/zeros:08
?
output/weights4:0output/weights4/Assign%output/weights4/Read/ReadVariableOp:0(2.output/weights4/Initializer/truncated_normal:08
w
output/biases4:0output/biases4/Assign$output/biases4/Read/ReadVariableOp:0(2"output/biases4/Initializer/zeros:08"? 
	variables? ? 
?
layer_1/weights1:0layer_1/weights1/Assign&layer_1/weights1/Read/ReadVariableOp:0(2/layer_1/weights1/Initializer/truncated_normal:08
{
layer_1/biases1:0layer_1/biases1/Assign%layer_1/biases1/Read/ReadVariableOp:0(2#layer_1/biases1/Initializer/zeros:08
?
layer_2/weights2:0layer_2/weights2/Assign&layer_2/weights2/Read/ReadVariableOp:0(2/layer_2/weights2/Initializer/truncated_normal:08
{
layer_2/biases2:0layer_2/biases2/Assign%layer_2/biases2/Read/ReadVariableOp:0(2#layer_2/biases2/Initializer/zeros:08
?
layer_3/weights3:0layer_3/weights3/Assign&layer_3/weights3/Read/ReadVariableOp:0(2/layer_3/weights3/Initializer/truncated_normal:08
{
layer_3/biases3:0layer_3/biases3/Assign%layer_3/biases3/Read/ReadVariableOp:0(2#layer_3/biases3/Initializer/zeros:08
?
output/weights4:0output/weights4/Assign%output/weights4/Read/ReadVariableOp:0(2.output/weights4/Initializer/truncated_normal:08
w
output/biases4:0output/biases4/Assign$output/biases4/Read/ReadVariableOp:0(2"output/biases4/Initializer/zeros:08
?
train/beta1_power:0train/beta1_power/Assign'train/beta1_power/Read/ReadVariableOp:0(2-train/beta1_power/Initializer/initial_value:0
?
train/beta2_power:0train/beta2_power/Assign'train/beta2_power/Read/ReadVariableOp:0(2-train/beta2_power/Initializer/initial_value:0
?
train/layer_1/weights1/Adam:0"train/layer_1/weights1/Adam/Assign1train/layer_1/weights1/Adam/Read/ReadVariableOp:0(2/train/layer_1/weights1/Adam/Initializer/zeros:0
?
train/layer_1/weights1/Adam_1:0$train/layer_1/weights1/Adam_1/Assign3train/layer_1/weights1/Adam_1/Read/ReadVariableOp:0(21train/layer_1/weights1/Adam_1/Initializer/zeros:0
?
train/layer_1/biases1/Adam:0!train/layer_1/biases1/Adam/Assign0train/layer_1/biases1/Adam/Read/ReadVariableOp:0(2.train/layer_1/biases1/Adam/Initializer/zeros:0
?
train/layer_1/biases1/Adam_1:0#train/layer_1/biases1/Adam_1/Assign2train/layer_1/biases1/Adam_1/Read/ReadVariableOp:0(20train/layer_1/biases1/Adam_1/Initializer/zeros:0
?
train/layer_2/weights2/Adam:0"train/layer_2/weights2/Adam/Assign1train/layer_2/weights2/Adam/Read/ReadVariableOp:0(2/train/layer_2/weights2/Adam/Initializer/zeros:0
?
train/layer_2/weights2/Adam_1:0$train/layer_2/weights2/Adam_1/Assign3train/layer_2/weights2/Adam_1/Read/ReadVariableOp:0(21train/layer_2/weights2/Adam_1/Initializer/zeros:0
?
train/layer_2/biases2/Adam:0!train/layer_2/biases2/Adam/Assign0train/layer_2/biases2/Adam/Read/ReadVariableOp:0(2.train/layer_2/biases2/Adam/Initializer/zeros:0
?
train/layer_2/biases2/Adam_1:0#train/layer_2/biases2/Adam_1/Assign2train/layer_2/biases2/Adam_1/Read/ReadVariableOp:0(20train/layer_2/biases2/Adam_1/Initializer/zeros:0
?
train/layer_3/weights3/Adam:0"train/layer_3/weights3/Adam/Assign1train/layer_3/weights3/Adam/Read/ReadVariableOp:0(2/train/layer_3/weights3/Adam/Initializer/zeros:0
?
train/layer_3/weights3/Adam_1:0$train/layer_3/weights3/Adam_1/Assign3train/layer_3/weights3/Adam_1/Read/ReadVariableOp:0(21train/layer_3/weights3/Adam_1/Initializer/zeros:0
?
train/layer_3/biases3/Adam:0!train/layer_3/biases3/Adam/Assign0train/layer_3/biases3/Adam/Read/ReadVariableOp:0(2.train/layer_3/biases3/Adam/Initializer/zeros:0
?
train/layer_3/biases3/Adam_1:0#train/layer_3/biases3/Adam_1/Assign2train/layer_3/biases3/Adam_1/Read/ReadVariableOp:0(20train/layer_3/biases3/Adam_1/Initializer/zeros:0
?
train/output/weights4/Adam:0!train/output/weights4/Adam/Assign0train/output/weights4/Adam/Read/ReadVariableOp:0(2.train/output/weights4/Adam/Initializer/zeros:0
?
train/output/weights4/Adam_1:0#train/output/weights4/Adam_1/Assign2train/output/weights4/Adam_1/Read/ReadVariableOp:0(20train/output/weights4/Adam_1/Initializer/zeros:0
?
train/output/biases4/Adam:0 train/output/biases4/Adam/Assign/train/output/biases4/Adam/Read/ReadVariableOp:0(2-train/output/biases4/Adam/Initializer/zeros:0
?
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign1train/output/biases4/Adam_1/Read/ReadVariableOp:0(2/train/output/biases4/Adam_1/Initializer/zeros:0*?
serving_default?
3
input*
input/Placeholder:0?????????	0
earnings$
output/Relu:0?????????tensorflow/serving/predict