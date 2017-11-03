-- @author Qi Hu, Sachin Mehta

local nn = require 'nn'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true



----------------------------------------------------------------------
print '==> define parameters'
--~ 
local classes = opt.classes

-- shortcuts for layers -------------------
local Convolution = cudnn.SpatialConvolution --cudnn
local DeConvolution = cudnn.SpatialFullConvolution--cudnn
local Avg = cudnn.SpatialAveragePooling--cudnn
local ReLU = nn.RReLU --cudnn.ReLU --cudnn
local Max = cudnn.SpatialMaxPooling
local UnPool = cudnn.SpatialMaxUnpooling
local SBatchNorm = nn.SpatialBatchNormalization
local Dropout = nn.SpatialDropout
local Dilated = nn.SpatialDilatedConvolution

local iChannels


--- -----
-- Function to create shortcut connection 
-- @function [parent=#MResNet] shortcutConnectionDec
function shortcutConnectionDec(nInputPlane, nOutputPlane, stride)
     -- 1x1 de-convolution
     return nn.Sequential()
        :add(DeConvolution(nInputPlane, nOutputPlane, 1, 1,stride,stride,0,0, stride - 1, stride -1))
        :add(SBatchNorm(nOutputPlane))
end

--- ------
-- Function to create multi-scale residual block
-- @function [parent=#MResNet] multiScaleResidualBlockDec

function multiScaleResidualBlockDec(n, stride)
  local nInputPlane = iChannels
  local mul = 2
  iChannels = n / mul
  depth_layer = n/2

  local s = nn.Sequential()
  s:add(DeConvolution(nInputPlane,depth_layer,1,1,stride,stride,0,0, stride - 1, stride -1))
  s:add(SBatchNorm(depth_layer))
  s:add(ReLU(1/8, 1/3, true))
  s:add(Convolution(depth_layer,depth_layer,3,3,1,1,1,1))
  s:add(SBatchNorm(depth_layer))
  

  return nn.Sequential()
     :add(nn.ConcatTable()
        :add(s)
        :add(shortcutConnectionDec(nInputPlane, iChannels, stride)))
     :add(nn.CAddTable(true))
     :add(ReLU(1/8, 1/3, true))
end

--- ------
-- Function to load the pretrained encoder (imagenet)
-- @function [parent=#MResNet] encoder

function encoder()
  enc = torch.load(opt.encModel)
  --drop the fully connected layers
  enc.modules[9] = nil
  enc.modules[10] = nil
  enc.modules[11] = nil
  return enc
end

--- --------
---- Function to create the decoder
-- @function [parent=#MResNet] decoder

function decoder()
  dropout_ratio = 0.2
  dec = nn.Sequential()
  iChannels = 512
  dec:add(multiScaleResidualBlockDec(512, 2))
  dec:add(Dropout(dropout_ratio))
  dec:add(multiScaleResidualBlockDec(512, 1))
  dec:add(Dropout(dropout_ratio))
  dec:add(multiScaleResidualBlockDec(256, 2))
  dec:add(Dropout(dropout_ratio))
  dec:add(multiScaleResidualBlockDec(256, 1))
  dec:add(Dropout(dropout_ratio))
  dec:add(multiScaleResidualBlockDec(128, 2))
  dec:add(Dropout(dropout_ratio))
  dec:add(multiScaleResidualBlockDec(128, 1))
  dec:add(Dropout(dropout_ratio))
  
  dec:add(DeConvolution(64,64,1,1,2,2,0,0, 1, 1))
  dec:add(Convolution(64, 64, 3, 3, 1, 1, 1, 1))
  dec:add(SBatchNorm(64))
  dec:add(ReLU(1/8, 1/3, true))
  dec:add(Dropout(dropout_ratio))
  
  dec:add(DeConvolution(64,64,1,1,2,2,0,0, 1, 1))
  dec:add(Convolution(64, 64, 3, 3, 1, 1, 1, 1))
  dec:add(SBatchNorm(64))
  dec:add(ReLU(1/8, 1/3, true))
  dec:add(Dropout(dropout_ratio))
  
  local function ConvInit(name)
    for k,v in pairs(dec:findModules(name)) do
    local n = v.kW*v.kH*v.nOutputPlane
    v.weight:normal(0,math.sqrt(2/n))
    if cudnn.version >= 4000 then
      v.bias = nil
      v.gradBias = nil
     else
      v.bias:zero()
     end
    end
  end
  local function BNInit(name)
    for k,v in pairs(dec:findModules(name)) do
     v.weight:fill(1)
     v.bias:zero()
    end
  end

  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')
  ConvInit('cudnn.SpatialFullConvolution')
  ConvInit('nn.SpatialFullConvolution')
  ConvInit('nn.SpatialDilatedConvolution')
  BNInit('cudnn.SpatialBatchNormalization')
  BNInit('nn.SpatialBatchNormalization')
  return dec
end

modelTrain = opt.modelType
local model
print('Finetuning M-Plain')
model = nn.Sequential()
model:add(encoder())
model:add(decoder())

model:add(Convolution(64,classes,3,3,1,1,1,1))

model:cuda()

return model
