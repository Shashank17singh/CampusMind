import { execSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

const isInsideFrontend = existsSync(resolve(process.cwd(), 'src/main.tsx'))
const rootDir = isInsideFrontend ? resolve(process.cwd(), '..') : process.cwd()

console.log(`Building CampusMind frontend (working directory: ${process.cwd()})...`)

if (isInsideFrontend) {
  execSync('npm run build', { stdio: 'inherit' })
} else {
  execSync('npm --prefix frontend run build', { stdio: 'inherit' })
  mkdirSync(resolve(rootDir, 'dist'), { recursive: true })
  cpSync(resolve(rootDir, 'frontend/dist'), resolve(rootDir, 'dist'), { recursive: true })
}

console.log('Build completed successfully.')
