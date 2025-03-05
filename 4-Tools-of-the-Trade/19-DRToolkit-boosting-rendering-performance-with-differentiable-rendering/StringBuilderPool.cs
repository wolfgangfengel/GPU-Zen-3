using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hlsl2Python
{
    internal sealed class StringBuilderPool
    {
        public bool IsDebugMode { get; set; } = false;

        public StringBuilder Alloc()
        {
            StringBuilder sb;
            if (m_Queue.Count > 0) {
                sb = m_Queue.Dequeue();
                if (IsDebugMode) {
                    m_Records.Remove(sb);
                }
                if (sb.Length > 0)
                    sb.Length = 0;
            }
            else {
                sb = new StringBuilder();
            }
            return sb;
        }
        public void Recycle(StringBuilder sb)
        {
            if (IsDebugMode) {
                string trace = Environment.StackTrace;
                if (m_Records.TryGetValue(sb, out var rec)) {
                    Console.WriteLine("[===pool conflict===]");
                    Console.WriteLine("===this===");
                    Console.WriteLine(trace);
                    Console.WriteLine("===other===");
                    Console.WriteLine(rec);
                    return;
                }
                else {
                    m_Records.Add(sb, trace);
                }
            }
            m_Queue.Enqueue(sb);
        }

        private Queue<StringBuilder> m_Queue = new Queue<StringBuilder>();
        private Dictionary<StringBuilder, string> m_Records = new Dictionary<StringBuilder, string>();
    }
}
