# AutoClass

MindFormers大模型套件提供了AutoClass类，包含AutoConfig、AutoModel、AutoTokenizer、AutoProcessor4个便捷高阶接口，方便用户调用套件中已封装的API接口。上述4类提供了相应领域模型的ModelConfig、Model、Tokenzier、Processor的实例化功能。

![输入图片说明](https://foruda.gitee.com/images/1686128219487920380/00f18fec_9324149.png)

| AutoClass类   | from_pretrained属性（实例化功能） | from_config属性（实例化功能） | save_pretrained（保存配置功能） |
| ------------- | --------------------------------- | ----------------------------- | :-----------------------------: |
| AutoConfig    | √                                 | ×                             |                √                |
| AutoModel     | √                                 | √                             |                √                |
| AutoProcessor | √                                 | ×                             |                √                |
| AutoTokenizer | √                                 | ×                             |                √                |
