<template>
  <div class="home">
    <el-steps :active="active" align-center finish-status="success" process-status="wait">
      <el-step title="选择检测模型"></el-step>
      <el-step title="上传检测内容"></el-step>
      <el-step title="查看检测报告"></el-step>
    </el-steps>
    <div v-if="active===1" type="algin-center">
      <div class="select">
        <h3>选择检测模型的类型:</h3>
        <el-radio v-model="type" label="special">专用型</el-radio>
        <el-radio v-model="type" label="general">通用型</el-radio>
      </div>
    </div>

    <div v-if="active===2&&type==='special'">
      <div class="upload">
        <div class="file-preview">
          <div class="forPreview_special" v-for="(item,index) in dialogspecialUrl" :key="item">
            <img class="specialFile" src="../assets/images/audio.png" alt="" style="width:200px;height: 200px;margin-right:10px">
            <audio width="200px" height="200px" style="display: inline-block" controls="controls" :src="item"> 您的浏览器不支持音频播放</audio>
            <img class="delete" src="../assets/images/close.png" style="width:20px;height:20px;background:rgba(0, 0, 0, .6);diaplay:inline-flex;" @click="forkspecial(index)">
          </div>
          <el-upload
            action = ' '
            :limit = "10"
            ref ="upload"
            class ="special"
            multiple
            :on-progress="uploadspecialProcess"
            :on-success="handlespecialSuccess"
            :before-upload="() => {}"
            :on-change="beforeUploadspecial"
            :show-file-list="false"
            :auto-upload="false"
            accept = ".flac">
            <div class=" file-drop-zone" v-if="!specialFlag && dialogspecialUrl.length == 0">
              <div class="el-upload__text">点击上传音频</div>
              <div class="el-upload__tip" slot="tip">只能上传flac格式，且不超过500kb</div>
            </div>
            <el-progress
              v-if="specialFlag == true"
              type="circle"
              :percentage="specialUploadPercent"
              style="margin-top:30px;position: relative;top: -15px;">
            </el-progress>
            <div style="margin-top: 15px">
              <el-button type = "primary" style="width:100%">批量上传</el-button>
            </div>
          </el-upload>
        </div>
      </div>
    </div>

    <div v-if="active===2&&type==='general'">
      <div class="upload">
        <div class="file-preview">
          <div class="forPreview_special" v-for="(item,index) in dialoggeneralUrl" :key="item">
            <img class="specialFile" src="../assets/images/audio.png" alt="" style="width:200px;height: 200px;margin-right:10px">
            <audio width="200px" height="200px" style="display: inline-block" controls="controls" :src="item"> 您的浏览器不支持音频播放</audio>
            <img class="delete" src="../assets/images/close.png" style="width:20px;height:20px;background:rgba(0, 0, 0, .6);diaplay:inline-flex;" @click="forkspecial(index)">
          </div>
          <el-upload
            action = ' '
            :limit = "10"
            ref ="upload"
            class ="general"
            multiple
            :on-progress="uploadgeneralProcess"
            :on-success="handlegeneralSuccess"
            :before-upload="() => {}"
            :on-change="beforeUploadgeneral"
            :show-file-list="false"
            :auto-upload="false"
            accept = ".flac">
            <div class=" file-drop-zone" v-if="!generalFlag && dialoggeneralUrl.length == 0">
              <div class="el-upload__text">点击上传音频</div>
              <div class="el-upload__tip" slot="tip">只能上传flac格式，且不超过500kb</div>
            </div>
            <el-progress
              v-if="generalFlag == true"
              type="circle"
              :percentage="generalUploadPercent"
              style="margin-top:30px;position: relative;top: -15px;">
            </el-progress>
            <div style="margin-top: 15px">
              <el-button type = "primary" style="width:100%">批量上传</el-button>
            </div>
          </el-upload>
        </div>
      </div>
    </div>

    <div class="report" v-if="active===3">
      <div class="box-card total">
        <!-- 饼状图 -->
        <div class="echart" id="mychart" style="width:100%; height:300px;"></div>
        <div slot="header" class="clearfix">
          <span>检测共计{{ reports.length }}个文件, 其中伪造音频{{ fakeFile ? fakeFile.length : "0" }}个, 真实音频{{ trueFile ? trueFile.length : "0" }}个。</span>
        </div>
      </div>

      <div class="detail">
        <div :gutter="12">
          <el-col
            v-for="report in reports"
            :key="reports.indexOf(report) + 1"
            :span="8">
            <div class="box-card">
              <div class="filename">{{report.nameOf}}</div>
              <div v-if="report.state === 0">
                该音频有{{ String(report.confidence) + "%" }}的概率是真实的。
              </div>
              <div v-else>
                该音频有{{ report.confidence + "%" }}的概率是伪造的。
              </div>
            </div>
          </el-col>
        </div>
      </div>
    </div>

    <div style="text-align: center;padding:10px">
      <el-button type="primary" @click="pre" v-if="active===2||active===3">上一步</el-button>
      <el-button type="primary" @click="next" v-if="active===1||active===2">下一步</el-button>
      <el-button type="primary" @click="next" v-else>返回</el-button>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import * as echarts from 'echarts'
import { Loading } from 'element-ui'

export default {
  name: 'home',
  data () {
    return {
      active: 1,
      type: 'special',
      // 专用
      specialFlag: false,
      specialUploadPercent: '',
      isShowUploadspecial: false,
      dialogspecialUrl: [],
      specialList: [],
      specialForm: {
        showspecialPath: ''
      },
      // 通用
      generalFlag: false,
      generalUploadPercent: 0,
      isShowUploadgeneral: false,
      dialoggeneralUrl: [],
      generalList: [],
      generalForm: {
        showgeneralPath: ''
      },
      reports:[ 
        {
          "confidence": 100.0,
          "nameOf": "special_audio1.flac",
          "state": 1
        },
        {
          "confidence": 98.0,
          "nameOf": "special_audio2.flac",
          "state": 0
        }],
      myChart:{}
    }
  },
  computed: {
    trueFile:function() { 
      return this.reports.filter((item) => item.state === 0)
    },
    fakeFile:function() {
      return this.reports.filter((item) => item.state === 1)
    },
  },
  methods: {
    // 提示框
    handleClose () {},

    next () {
      if (this.active++ > 2) this.active = 1
      if (this.active == 2) {
      }
      if (this.active == 3) {
        if(this.type == 'special') {
          this.uploadspecialFile()
        }else {
          this.uploadgeneralFile()
        }
        this.openFullScreen()
        // 开始绘图
        // this.$nextTick(function () {
        //   this.getPie()
        // })
      }
    },
    pre () {
      if (this.active-- < 1) this.active = 1
    },
    // 1.专用检测模型
    
    // 删除文件
    forkspecial (index) {
      this.dialogspecialUrl.splice(index, 1);
      this.specialList.splice(index,1)
    },
    // 传文件之前的钩子
    beforeUploadspecial (fileObj,specialList) {
      var file = fileObj.raw
      var fileSize = file.size / 1024 / 1024 < 500
      if (['audio/flac', 'audio/mpeg'].indexOf(file.type) === -1) {
        console.log(`请上传正确的音频格式: ${file.type}`)
        return false
      }
      if (!fileSize) {
        console.log('音频大小不能超过500MB')
        return false
      }
      this.isShowUploadspecial = true
      this.specialFlag = false
      this.specialUploadPercent = 0
      this.dialogspecialUrl.push(URL.createObjectURL(file))   
      this.specialList.push(fileObj)
    },

    // 上传文件
    uploadspecialFile () {
      const formData = new FormData()
      formData.append('type', this.type)
      formData.append('number', this.specialList.length)
      this.specialList.forEach((file, index) => {
        formData.append('filename[' + index + ']', file.name)
        formData.append('files[' + index + ']', file.raw)
      })
      const path = '/api/special'
      axios.post(path, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }).then(res => {
        this.reports = res.data
        this.getPie()
      }).catch((error) => {
        console.log('error', error)
        this.getPie()
      })
    },
    // 上传成功回调
    handlespecialSuccess (res, file) {
      this.isShowUploadspecial = true
      this.specialFlag = false
      this.specialUploadPercent = 0
      // 后台上传地址
      if (res.Code === 0) {
        this.dialogspecialUrl = res.Data
        console.log("file",file)
      } else {
        console.log(res.Message)
      }
    },
    // 进度条
    uploadspecialProcess (event) {
      this.specialFlag = true
      this.specialUploadPercent = Math.floor(event.percent)
    },
    
    // 上传文件个数超过定义的数量
    handleExceed () {
      this.$message.warning('当前限制选择 10 个文件，请删除后继续上传')
    },
   
    // 2.通用检测模型
    // 删除文件
    forkgeneral (index) {
      this.dialoggeneralUrl.splice(index, 1);
      this.generalList.splice(index,1)
      },
    // 传文件之前的钩子
    beforeUploadgeneral (fileObj,generalList) {
      var file = fileObj.raw
      var fileSize = file.size / 1024 / 1024 < 500
      if (['audio/flac', 'audio/mpeg'].indexOf(file.type) === -1) {
        console.log(`请上传正确的音频格式: ${file.type}`)
        return false
      }
      if (!fileSize) {
        console.log('音频大小不能超过500MB')
        return false
      }
      this.isShowUploadgeneral = true
      this.generalFlag = false
      this.generalUploadPercent = 0
      this.dialoggeneralUrl.push(URL.createObjectURL(file))   
      this.generalList.push(fileObj)
    },

    // 上传文件
    uploadgeneralFile () {
      const formData = new FormData()
      formData.append('type', this.type)
      formData.append('number', this.generalList.length)
      this.generalList.forEach((file, index) => {
        formData.append('filename[' + index + ']', file.name)
        formData.append('files[' + index + ']', file.raw)
      })
      const path = '/api/general'
      axios.post(path, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }).then(res => {
        this.reports = res.data
        console.log(this.reports);
        this.getPie()
      }).catch((error) => {
        console.log('error', error)
        this.getPie()
      })
    },
    // 上传成功回调
    handlegeneralSuccess (res, file) {
      this.isShowUploadgeneral = true
      this.generalFlag = false
      this.generalUploadPercent = 0
      // 后台上传地址
      if (res.Code === 0) {
        this.dialoggeneralUrl = res.Data
        console.log("file",file)
      } else {
        console.log(res.Message)
      }
    },
    // 进度条
    uploadgeneralProcess (event) {
      this.generalFlag = true
      this.generalUploadPercent = Math.floor(event.percent)
    },

    // 上传文件个数超过定义的数量
    handleExceed () {
      this.$message.warning('当前限制选择 10 个文件，请删除后继续上传')
    },
 
    //图表
    getPie () {
      // 指定图表的配置项和数据
      let defaultOption = {
        title: {
          text: '检测结果',
          x: 'center',
          textStyle: { // 标题内容的样式
            color: '#000',
            fontStyle: 'normal',
            fontWeight: 100,
            fontSize: 16
          }
        },
        tooltip: {
          trigger: 'item',
          formatter: '{b}:{c} ({d}%)'
        },
        // 图例
        legend: {
          orient: 'vertical', // 图例的显示方式
          right: 10, // 图例出现的距离
          textStyle: {
            color: '#000',
            fontSize: 16
          },
          data: ['伪造', '非伪造']
        },
        // 饼图中各模块的颜色
        color: ['#32dadd', '#b6a2de'],
        // 饼图数据
        series: {
          type: 'pie',
          radius: '70%',
          center: ['50%', '50%'],
          data: [
            { name: '伪造', value: this.fakeFile.length },
            { name: '非伪造', value: this.trueFile.length }
          ],
          label: {
            show: false // 饼图上是否出现标注文字
          }

        }
      }
      let singleOption = {
        tooltip: {
          trigger: 'item',
          formatter: '{b}:{c} ({d}%)'
        },
        // 图例
        legend: {
          orient: 'vertical', // 图例的显示方式
          right: 10, // 图例出现的距离
          textStyle: {
            color: '#000',
            fontSize: 16
          },
          data: ['检测结果可信度', '']
        },
        // 饼图中各模块的颜色
        color: ['#32dadd', '#b6a2de'],
        // 饼图数据
        series: {
          type: 'pie',
          radius: '70%',
          center: ['50%', '50%'],
          data: [
            { name: '检测结果可信度', value: this.reports[0].confidence },
            { name: '', value: (100 - this.reports[0].confidence).toFixed(2) }
          ],
          label: {
            show: false // 饼图上是否出现标注文字
          }
        }
      }
      this.myChart = echarts.init(document.getElementById('mychart'))
      if(this.reports.length === 1){
        this.myChart.setOption(singleOption)
      }else{
        this.myChart.setOption(defaultOption)
      }
    },
    // 整页加载
    openFullScreen () {
      let loadingInstance = Loading.service();
        setTimeout(() => {
          this.$nextTick(() => { // 以服务的方式调用的 Loading 需要异步关闭
            loadingInstance.close();
          });
        }, 2000);
    },
  },
}
</script>
<style scoped>
  .select{
    font-size: 180%;
    text-align: center;
    margin: 100px auto;
    color: gray;
  }
  .upload {
    text-align: center;
    margin: 100px auto;
  }
 /deep/.el-upload{
    display: block;
    text-align: center;
    cursor: pointer;
    outline: 0;
  }
  .el-steps {
    margin: 20px -20px;
    overflow: hidden;
  }
  .el-icon-service {
    font-size: 60px;
    color: #C0C4CC;
    margin: 40px 0 16px;
    line-height: 50px;
  }
  .el-upload__text {
    color: #aaa;
    font-size: 1.6em;
    padding: 85px 10px;
    cursor: default;
  }
  .file-preview {
    border-radius: 5px;
    border: 1px solid #ddd;
    padding: 8px;
    width: 100%;
    margin-bottom: 5px;
    position: relative;
}
.file-drop-zone {
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.5;
    color: #212529;
    box-sizing: border-box;
    border: 1px dashed #aaa;
    border-radius: 4px;
    height: 100%;
    text-align: center;
    vertical-align: middle;
    margin: 12px 15px 12px 12px;
    padding: 5px;
}
.el-input {
    margin: 12px;
    width: 100%;
}
.input-with-select{
    background-color: #fff;
}
.el-input-group__append .el-button{
  color: #fff;
  background-color:lightgreen;
}
.img-list-item {
    position: relative;
    /* margin: auto; */
    display: inline-block;
    margin: 20px 50px;
}
.img-list-item .del-img {
    width: 20px;
    height: 20px;
    background: rgba(0, 0, 0, .6);
    background-image: url(../assets/images/close.png);
    background-size: 18px;
    background-repeat: no-repeat;
    background-position: 50%;
    position: absolute;
    top: 0;
    right: -1px;
}
.forPreview_special {
  /* display: flex;
  margin-bottom: 10px;
  align-items: center;
  justify-content: space-around; */
  display: inline-block;
  margin: 10px 50px;
  position: relative;
}
.forPreview_special .specialFile{
  display: block;
  margin: 10px;
}
.forPreview_special .delete{
  /* 相对父元素定位 */
  position:absolute;
  right: -1px;
  top: 10px;
}

/* report布局 */
.detail{
    width: 70%;
    margin: 0 auto;
}
.box-card{
  width: 575px;
  /* height: 61px; */
  display: inline-block;
  border: 1px rgba(128, 128, 128, 0.315) solid;
  border-radius: 10px;
  padding: 10px 5px;
  margin: 10px 0px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  background-color: #fff;
  word-wrap: break-word;
  word-break: normal;
}
.detail .box-card .filename{
  font-weight: normal;
  color: #aaa;
  font-size: 12px;
}
.total{
  padding: 10px;
  margin: 20px auto;
  display: block;
  width: 70%;
  height:auto;
  border-radius: 3px;
}
#chart{
  margin: 10px auto;
}
</style>
